#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
POST_PARALLEL_JOBS="${POST_PARALLEL_JOBS:-4}"
TEACHER_SEEDS=(43 44)
RECOVERY_SEEDS=(41 42 43)
STUDENT_SEEDS=(42 43 44)
TEMPERATURE=20
FKD_BATCH_SIZE=10

RANDOM_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_random_c200_labeler"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_random_c200_labeler}"
VAL_DIR="$val_dir/imagenet-nette/test"
mkdir -p "$EXP_ROOT/analysis" "$LOG_ROOT"

fail(){ echo "Random-C200 labeler experiment failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }

source_for(){
    local row="$1" teacher_seed="$2" recovery_seed="$3"
    if [[ "$row" == real ]]; then
        echo "$FACTORIAL_ROOT/real_sets/tseed${teacher_seed}_rseed${recovery_seed}"
    elif [[ "$row" == c1 ]]; then
        echo "$RANDOM_ROOT/tseed${teacher_seed}/synthetic/cic_t_c1_ipc10_rseed${recovery_seed}"
    else
        return 1
    fi
}
teacher_for(){
    local teacher_seed="$1"
    echo "$RANDOM_ROOT/tseed${teacher_seed}/models/random_c200_pseed42_tseed${teacher_seed}/ResNet18.pth"
}
mapping_for(){
    local teacher_seed="$1"
    echo "$RANDOM_ROOT/tseed${teacher_seed}/data/random_c200_pseed42/hierarchy.json"
}

echo "[1/4] Strict existing-asset preflight"
[[ "$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 3925 ]] \
    || fail "post-eval is not the full ImageNette test split"
for teacher_seed in "${TEACHER_SEEDS[@]}"; do
    teacher="$(teacher_for "$teacher_seed")"
    mapping="$(mapping_for "$teacher_seed")"
    [[ -f "$teacher" && -f "$mapping" ]] || fail "missing Random-C200 Teacher assets: tseed=$teacher_seed"
    mapping_valid="$(python -c "import json; q=json.load(open('$mapping')); m=q.get('fine_to_coarse',{}); print(q.get('kind')=='imagenette_balanced_random_subclasses' and int(q.get('subclasses_per_coarse',-1))==200 and int(q.get('num_pseudo_classes',-1))==2000 and len(m)==2000 and all(int(m[str(i)])==i//200 for i in range(2000)))")"
    [[ "$mapping_valid" == True ]] || fail "invalid Random-C200 mapping: tseed=$teacher_seed"
    for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
        for row in real c1; do
            source="$(source_for "$row" "$teacher_seed" "$recovery_seed")"
            [[ "$(find "$source" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 100 ]] \
                || fail "invalid source: row=$row tseed=$teacher_seed rseed=$recovery_seed"
            [[ "$(find "$source" -mindepth 1 -maxdepth 1 -type d | wc -l)" == 10 ]] \
                || fail "source does not have ten parent directories"
        done
    done
done

relabel_one(){
    local row="$1" teacher_seed="$2" recovery_seed="$3" gpu="$4"
    local source teacher mapping seed_root base final count archive
    source="$(source_for "$row" "$teacher_seed" "$recovery_seed")"
    teacher="$(teacher_for "$teacher_seed")"
    mapping="$(mapping_for "$teacher_seed")"
    seed_root="$EXP_ROOT/tseed${teacher_seed}"
    base="$seed_root/fkd/${row}__random200_rseed${recovery_seed}"
    final="${base}_bs10_ipc10"
    mkdir -p "$seed_root/fkd"
    count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    if (( count > 0 )); then
        archive="${final}.partial_$(date +%Y%m%d_%H%M%S)"
        [[ ! -e "$archive" ]] || return 1
        mv "$final" "$archive"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$source" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" --teacher-model-name ResNet18 \
        --teacher-num-classes 2000 --teacher-mapping "$mapping" \
        --marginalize-temperature 20 --gpu 0 --batch-size 10 --workers "$WORKERS" \
        --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette \
        --epochs 300 --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel_${row}_t${teacher_seed}_r${recovery_seed}.log" 2>&1 || return 1
    [[ "$(find "$final" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]]
}

echo "[2/4] Random-C200 relabel: two concurrent streams"
pids=()
task=0
for teacher_seed in "${TEACHER_SEEDS[@]}"; do
    for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
        for row in real c1; do
            gpu="$GPU0"
            (( task % 2 )) && gpu="$GPU1"
            task=$((task + 1))
            relabel_one "$row" "$teacher_seed" "$recovery_seed" "$gpu" & pids+=("$!")
            if (( ${#pids[@]} == 2 )); then
                wait_jobs "${pids[@]}" || fail relabel
                pids=()
            fi
        done
    done
done
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail relabel

post_one(){
    local row="$1" teacher_seed="$2" recovery_seed="$3" student_seed="$4" gpu="$5"
    local source seed_root fkd result valid run_name
    source="$(source_for "$row" "$teacher_seed" "$recovery_seed")"
    seed_root="$EXP_ROOT/tseed${teacher_seed}"
    fkd="$seed_root/fkd/${row}__random200_rseed${recovery_seed}_bs10_ipc10"
    result="$seed_root/per_class/${row}__random200_rseed${recovery_seed}_sseed${student_seed}.json"
    run_name="random_c200_labeler_${row}_t${teacher_seed}_r${recovery_seed}_s${student_seed}"
    mkdir -p "$seed_root/per_class" "$seed_root/post_eval"
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and q.get('training_target')=='fkd_soft_label')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    [[ "$(find "$fkd" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] || return 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" \
        --fkd-path "$fkd" --mix-type cutmix --temperature 20 \
        --model ResNet18 --ipc 10 --exp-name "$run_name" \
        --original-data-path "$source" --output-dir "$seed_root/post_eval" \
        --batch-size 10 --epochs 300 --dataset-name imagenet-nette \
        --gradient-accumulation-steps 2 --cos --workers "$WORKERS" \
        --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005 \
        --eta-override 1 --train-seed "$student_seed" --persistent-workers \
        --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
        > "$LOG_ROOT/post_${row}_t${teacher_seed}_r${recovery_seed}_s${student_seed}.log" 2>&1
}

echo "[3/4] 36 post-eval cells, two processes per A100"
pids=()
task=0
for row in real c1; do
    for teacher_seed in "${TEACHER_SEEDS[@]}"; do
        for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
            for student_seed in "${STUDENT_SEEDS[@]}"; do
                gpu="$GPU0"
                (( task % 2 )) && gpu="$GPU1"
                task=$((task + 1))
                post_one "$row" "$teacher_seed" "$recovery_seed" "$student_seed" "$gpu" & pids+=("$!")
                if (( ${#pids[@]} == POST_PARALLEL_JOBS )); then
                    wait_jobs "${pids[@]}" || fail post_eval
                    pids=()
                fi
            done
        done
    done
done
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post_eval

echo "[4/4] Summarize and pair against C1/Random-C100 labelers"
PYTHONPATH="$ROOT/class_in_class:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/summarize_imagenette_random_c200_labeler.py" \
    --experiment-root "$EXP_ROOT" --random-root "$RANDOM_ROOT" \
    --factorial-root "$FACTORIAL_ROOT" --output "$EXP_ROOT/analysis/summary.json" \
    > "$LOG_ROOT/summarize.log" 2>&1 || fail summarize

echo "Complete: $EXP_ROOT/analysis/summary.json"
