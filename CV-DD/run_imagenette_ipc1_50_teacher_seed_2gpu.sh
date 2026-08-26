#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 43 or 44}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
POST_PARALLEL_JOBS="${POST_PARALLEL_JOBS:-4}"
[[ "$POST_PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "POST_PARALLEL_JOBS must be positive" >&2; exit 1; }
RSEEDS=(41 42 43); SSEEDS=(42 43 44); IPCS=(1 50)
RANDOM_ROOT="${RANDOM_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
EXP_ROOT="${EXP_ROOT:-$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"; REAL_ROOT="$val_dir/imagenet-nette"
TEACHER_ROOT="$RANDOM_ROOT/tseed${TEACHER_SEED}"
TEACHER_C1="$TEACHER_ROOT/models/random_c1_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
TEACHER_R100="$TEACHER_ROOT/models/random_c100_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
MAP_C1="$TEACHER_ROOT/data/random_c1_pseed42/hierarchy.json"
MAP_R100="$TEACHER_ROOT/data/random_c100_pseed42/hierarchy.json"
PATCH_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/patches/c1_ipc50"
SYN_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/synthetic"
SOURCE_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/sources"
FKD_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/fkd"
POST_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/post_eval"
RESULT_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/per_class"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_ipc1_50_main_table/tseed${TEACHER_SEED}}"
mkdir -p "$PATCH_ROOT" "$SYN_ROOT" "$SOURCE_ROOT" "$FKD_ROOT" "$POST_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

fail(){ echo "IPC1/50 Teacher-seed pipeline failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }

[[ -f "$TEACHER_C1" && -f "$TEACHER_R100" && -f "$MAP_C1" && -f "$MAP_R100" ]] \
    || fail "missing C1/Random100 Teacher assets for seed$TEACHER_SEED"

echo "[1/6] Generate/reuse 50 patches per class for C1 Teacher"
patch_count=0; [[ -d "$PATCH_ROOT/medium" ]] && patch_count="$(find "$PATCH_ROOT/medium" -type f -name '*.jpg' | wc -l)"
if ! (( patch_count == 500 )) || ! python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$PATCH_ROOT" --classes 10 --patches-per-class 50 --image-size 224 \
    > "$LOG_ROOT/patch_validate.log" 2>&1; then
    if (( patch_count > 0 )); then
        archive="${PATCH_ROOT}.invalid_$(date +%Y%m%d_%H%M%S)"
        mv "$PATCH_ROOT" "$archive"; mkdir -p "$PATCH_ROOT"
    fi
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/generate_patches.py" \
        --data-dir "$REAL_ROOT" --teacher "$TEACHER_C1" --teacher-num-classes 10 \
        --teacher-mapping "$MAP_C1" --teacher-architecture torchvision \
        --num-classes 10 --patches-per-class 50 --candidate-images 100 \
        --crops-per-image 5 --image-size 224 --normalization imagenet \
        --scoring-batch-size 256 --crop-workers 16 --output-dir "$PATCH_ROOT/medium" --seed 42 \
        > "$LOG_ROOT/patch.log" 2>&1 || fail "patch generation; see $LOG_ROOT/patch.log"
fi
python "$ROOT/class_in_class/validate_cvdd_patch_tree.py" \
    --patch-dir "$PATCH_ROOT" --classes 10 --patches-per-class 50 --image-size 224 \
    > "$LOG_ROOT/patch_validate.log" 2>&1 || fail "patch validation"

recover_one(){
    local recovery_seed="$1" gpu="$2"
    local exp="c1_ipc50_rseed${recovery_seed}"
    local output="$SYN_ROOT/$exp"
    local marker="$output/.protocol" count=0 expected archive
    expected="teacher=$(sha256sum "$TEACHER_C1"|awk '{print $1}'):mapping=$(sha256sum "$MAP_C1"|awk '{print $1}'):patch=$(find "$PATCH_ROOT/medium" -type f -name '*.jpg' -print0|sort -z|xargs -0 sha256sum|sha256sum|awk '{print $1}'):rseed=$recovery_seed:ipc=50:iter=4000:lr=0.1:r_bn=0.01"
    [[ -d "$output" ]] && count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    if [[ "$count" == 500 && -f "$marker" && "$(tr -d '[:space:]' < "$marker")" == "$expected" ]]; then return; fi
    if (( count > 0 )); then
        archive="${output}.partial_$(date +%Y%m%d_%H%M%S)"
        mv "$output" "$archive"
        echo "Archived partial IPC50 recovery; full restart required for RNG integrity: $archive"
    fi
    mkdir -p "$output"; printf '%s\n' "$expected" > "$marker"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/recover/recover.py" --exp-name "$exp" --apply-data-augmentation \
        --dataset-name imagenet-nette --batch-size 10 --syn-data-path "$SYN_ROOT" \
        --patch-dir "$PATCH_ROOT" --model-pool-dir "$(dirname "$TEACHER_C1")" \
        --pretrained-model-type offline --model-setting 0 --sre2l-model ResNet18 \
        --teacher-num-classes 10 --teacher-mapping "$MAP_C1" --voter-type equal \
        --selected-size 1 --lr 0.1 --iteration 4000 --r-bn 0.01 --store-best-images \
        --ipc-start 0 --ipc-end 50 --initialisation-method Patches --patch-diff medium \
        --seed "$recovery_seed" > "$LOG_ROOT/recover_r${recovery_seed}.log" 2>&1 \
        || return 1
    count="$(find "$output" -type f -name '*.jpg' | wc -l)"
    [[ "$count" == 500 ]] || { echo "IPC50 recovery incomplete: r=$recovery_seed count=$count" >&2; return 1; }
}

echo "[2/6] Full fresh IPC50 Recovery for three recovery seeds"
recover_one 41 "$GPU0" & p1=$!
recover_one 42 "$GPU1" & p2=$!
wait_jobs "$p1" "$p2" || fail "recovery seeds41/42"
recover_one 43 "$GPU0" || fail "recovery seed43"

source_for(){
    local row="$1" ipc="$2" recovery_seed="$3"
    case "$row" in
        real) echo "$SOURCE_ROOT/real_ipc${ipc}_rseed${recovery_seed}" ;;
        c1) echo "$SOURCE_ROOT/c1_ipc${ipc}_rseed${recovery_seed}" ;;
        *) return 1 ;;
    esac
}
teacher_for(){ case "$1" in c1) echo "$TEACHER_C1" ;; random100) echo "$TEACHER_R100" ;; *) return 1 ;; esac; }
mapping_for(){ case "$1" in c1) echo "$MAP_C1" ;; random100) echo "$MAP_R100" ;; *) return 1 ;; esac; }
heads_for(){ case "$1" in c1) echo 10 ;; random100) echo 1000 ;; *) return 1 ;; esac; }

echo "[3/6] Materialize IPC1 synthetic reuse and IPC-specific independent real subsets"
for recovery_seed in "${RSEEDS[@]}"; do
    existing_ipc10="$TEACHER_ROOT/synthetic/cic_t_c1_ipc10_rseed${recovery_seed}"
    python "$ROOT/class_in_class/prepare_imagefolder_ipc_subset.py" \
        --source "$existing_ipc10" --output "$(source_for c1 1 "$recovery_seed")" --ipc 1 \
        > "$LOG_ROOT/source_c1_ipc1_r${recovery_seed}.log" 2>&1
    # IPC50 source is the freshly recovered tree; expose it through a stable link.
    c1_ipc50_source="$(source_for c1 50 "$recovery_seed")"
    if [[ ! -e "$c1_ipc50_source" ]]; then
        ln -s "$SYN_ROOT/c1_ipc50_rseed${recovery_seed}" "$c1_ipc50_source"
    fi
    for ipc in "${IPCS[@]}"; do
        subset_seed=$((52000000 + TEACHER_SEED * 100003 + recovery_seed * 1009 + ipc * 10000019))
        python "$ROOT/class_in_class/prepare_imagenette_random_real_ipc10.py" \
            --source-train "$REAL_ROOT/train" --output-dir "$(source_for real "$ipc" "$recovery_seed")" \
            --seed "$subset_seed" --images-per-class "$ipc" --repair-invalid-output \
            > "$LOG_ROOT/source_real_ipc${ipc}_r${recovery_seed}.log" 2>&1
    done
done

relabel_one(){
    local ipc="$1" row="$2" column="$3" recovery_seed="$4" gpu="$5"
    local source teacher mapping heads base final expected_count count=0
    source="$(source_for "$row" "$ipc" "$recovery_seed")"; teacher="$(teacher_for "$column")"
    mapping="$(mapping_for "$column")"; heads="$(heads_for "$column")"
    base="$FKD_ROOT/ipc${ipc}_${row}__${column}_rseed${recovery_seed}"; final="${base}_bs10_ipc${ipc}"
    expected_count=$((300*ipc)); [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar'|wc -l)"
    (( count == expected_count )) && return
    if (( count > 0 )); then mv "$final" "${final}.partial_$(date +%Y%m%d_%H%M%S)"; fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" --syn-data-path "$source" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" --teacher-model-name ResNet18 \
        --teacher-num-classes "$heads" --teacher-mapping "$mapping" \
        --marginalize-temperature 20 --gpu 0 --batch-size 10 --workers "$WORKERS" \
        --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel_ipc${ipc}_${row}__${column}_r${recovery_seed}.log" 2>&1 || return 1
    count="$(find "$final" -type f -name 'batch_*.tar'|wc -l)"
    [[ "$count" == "$expected_count" ]] || return 1
}

echo "[4/6] Relabel all IPC1/50 soft cells"
pids=(); task_id=0
for ipc in "${IPCS[@]}"; do for recovery_seed in "${RSEEDS[@]}"; do for row in real c1; do for column in c1 random100; do
    gpu="$GPU0"; (( task_id % 2 == 1 )) && gpu="$GPU1"; task_id=$((task_id+1))
    relabel_one "$ipc" "$row" "$column" "$recovery_seed" "$gpu" & pids+=("$!")
    if (( ${#pids[@]} == 2 )); then wait_jobs "${pids[@]}" || fail "relabel"; pids=(); fi
done; done; done; done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail "relabel"; fi

post_one(){
    local ipc="$1" row="$2" column="$3" recovery_seed="$4" student_seed="$5" gpu="$6"
    local source result fkd args=()
    source="$(source_for "$row" "$ipc" "$recovery_seed")"
    result="$RESULT_ROOT/ipc${ipc}_${row}__${column}_rseed${recovery_seed}_sseed${student_seed}.json"
    if [[ -f "$result" ]]; then
        expected_target=fkd_soft_label; [[ "$column" == hard ]] && expected_target=hard_coarse_label
        valid="$(python -c "import json,os; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and os.path.realpath(q.get('validation_dir',''))==os.path.realpath('$VAL_DIR') and q.get('training_target')=='$expected_target')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    if [[ "$column" == hard ]]; then args+=(--hard-label); else
        fkd="$FKD_ROOT/ipc${ipc}_${row}__${column}_rseed${recovery_seed}_bs10_ipc${ipc}"
        args+=(--fkd-path "$fkd" --mix-type cutmix --temperature 20)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" "${args[@]}" --model ResNet18 --ipc "$ipc" \
        --exp-name "ipc${ipc}_${row}__${column}_t${TEACHER_SEED}_r${recovery_seed}_s${student_seed}" \
        --original-data-path "$source" --output-dir "$POST_ROOT" --batch-size 10 --epochs 300 \
        --dataset-name imagenet-nette --gradient-accumulation-steps 2 --cos --workers "$WORKERS" \
        --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005 --eta-override 2 \
        --train-seed "$student_seed" --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" > "$LOG_ROOT/post_ipc${ipc}_${row}__${column}_r${recovery_seed}_s${student_seed}.log" 2>&1
}

echo "[5/6] Post-eval 12 cells across IPC1/50"
pids=(); task_id=0
for ipc in "${IPCS[@]}"; do for recovery_seed in "${RSEEDS[@]}"; do for row in real c1; do for column in hard c1 random100; do for student_seed in "${SSEEDS[@]}"; do
    gpu="$GPU0"; (( task_id % 2 == 1 )) && gpu="$GPU1"; task_id=$((task_id+1))
    post_one "$ipc" "$row" "$column" "$recovery_seed" "$student_seed" "$gpu" & pids+=("$!")
    if (( ${#pids[@]} == POST_PARALLEL_JOBS )); then wait_jobs "${pids[@]}" || fail "post-eval"; pids=(); fi
done; done; done; done; done
if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail "post-eval"; fi
echo "[6/6] Teacher seed$TEACHER_SEED complete"
