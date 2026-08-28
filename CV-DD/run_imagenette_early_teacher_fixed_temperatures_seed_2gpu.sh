#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 43 or 44}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RSEEDS=(41 42); SSEEDS=(42 43)
MODES=(t8 t46 t100 t200)
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
EXISTING_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_downstream"
SEED_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}"
PLAN="$SEED_ROOT/selection_early.tsv"
FKD_ROOT="$SEED_ROOT/fkd"; RESULT_ROOT="$SEED_ROOT/per_class"; POST_ROOT="$SEED_ROOT/post_eval"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_early_teacher_fixed_temperatures/tseed${TEACHER_SEED}}"
VAL_DIR="$val_dir/imagenet-nette/test"
mkdir -p "$FKD_ROOT" "$RESULT_ROOT" "$POST_ROOT" "$LOG_ROOT"
fail(){ echo "Early Teacher fixed-temperature grid failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }
[[ -f "$PLAN" && "$(wc -l < "$PLAN")" == 18 ]] || fail "missing 18-checkpoint selection plan"

temperature_for(){
    case "$1" in t8) echo 8 ;; t46) echo 46 ;; t100) echo 100 ;; t200) echo 200 ;; *) return 1 ;; esac
}
source_for(){
    local source="$1" rseed="$2"
    if [[ "$source" == real ]]; then
        echo "$FACTORIAL_ROOT/real_sets/tseed${TEACHER_SEED}_rseed${rseed}"
    else
        echo "$EXISTING_ROOT/tseed${TEACHER_SEED}/synthetic/cic_t_c1_ipc10_rseed${rseed}"
    fi
}
mapping_for(){ echo "$TRAJECTORY_ROOT/tseed${TEACHER_SEED}/data/random_c${1}_pseed42/hierarchy.json"; }
heads_for(){ echo $((10 * $1)); }
tag_for(){ local c="$1" label="$2" epoch="$3" mode="$4"; echo "c${c}_${label}_e$(printf '%03d' "$epoch")_${mode}"; }

relabel_one(){
    local c="$1" label="$2" epoch="$3" teacher_view="$4" source="$5" mode="$6" rseed="$7" gpu="$8"
    local temperature tag mapping heads data base final count
    temperature="$(temperature_for "$mode")"; tag="$(tag_for "$c" "$label" "$epoch" "$mode")"
    mapping="$(mapping_for "$c")"; heads="$(heads_for "$c")"; data="$(source_for "$source" "$rseed")"
    base="$FKD_ROOT/${source}__${tag}_rseed${rseed}"; final="${base}_bs10_ipc10"
    count=0; [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    (( count > 0 )) && mv "$final" "${final}.partial_$(date +%Y%m%d_%H%M%S)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$data" --fkd-path "$base" --model-pool-dir "$teacher_view" \
        --teacher-model-name ResNet18 --teacher-num-classes "$heads" \
        --teacher-mapping "$mapping" --marginalize-temperature "$temperature" \
        --gpu 0 --batch-size 10 --workers "$WORKERS" --persistent-workers --prefetch-factor 4 \
        --dataset-name imagenet-nette --epochs 300 --fkd-seed 42 --seed 42 \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 \
        --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel_${source}__${tag}_r${rseed}.log" 2>&1 || return 1
    [[ "$(find "$final" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]]
}

echo "[1/2] Relabel 18 checkpoints at T=8/46/100/200"
pids=(); task=0
while IFS=$'\t' read -r c label epoch actual_val sd_z predicted teacher_view; do
    for source in real c1; do for mode in "${MODES[@]}"; do for rseed in "${RSEEDS[@]}"; do
        gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
        relabel_one "$c" "$label" "$epoch" "$teacher_view" "$source" "$mode" "$rseed" "$gpu" & pids+=("$!")
        if (( ${#pids[@]} == 2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
    done; done; done
done < "$PLAN"
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail relabel

post_one(){
    local c="$1" label="$2" epoch="$3" source="$4" mode="$5" rseed="$6" sseed="$7" gpu="$8"
    local temperature tag data fkd result valid
    temperature="$(temperature_for "$mode")"; tag="$(tag_for "$c" "$label" "$epoch" "$mode")"
    data="$(source_for "$source" "$rseed")"; fkd="$FKD_ROOT/${source}__${tag}_rseed${rseed}_bs10_ipc10"
    result="$RESULT_ROOT/${source}__${tag}_rseed${rseed}_sseed${sseed}.json"
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and q.get('training_target')=='fkd_soft_label')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" \
        --fkd-path "$fkd" --mix-type cutmix --temperature "$temperature" \
        --model ResNet18 --ipc 10 --exp-name "early_${source}__${tag}_t${TEACHER_SEED}_r${rseed}_s${sseed}" \
        --original-data-path "$data" --output-dir "$POST_ROOT" --batch-size 10 --epochs 300 \
        --dataset-name imagenet-nette --gradient-accumulation-steps 2 --cos --workers "$WORKERS" \
        --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005 --eta-override 1 \
        --train-seed "$sseed" --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" > "$LOG_ROOT/post_${source}__${tag}_r${rseed}_s${sseed}.log" 2>&1
}

echo "[2/2] Post-eval with two processes per A100"
pids=(); task=0
while IFS=$'\t' read -r c label epoch actual_val sd_z predicted teacher_view; do
    for source in real c1; do for mode in "${MODES[@]}"; do for rseed in "${RSEEDS[@]}"; do for sseed in "${SSEEDS[@]}"; do
        gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
        post_one "$c" "$label" "$epoch" "$source" "$mode" "$rseed" "$sseed" "$gpu" & pids+=("$!")
        if (( ${#pids[@]} == 4 )); then wait_jobs "${pids[@]}" || fail post; pids=(); fi
    done; done; done; done
done < "$PLAN"
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post
echo "Fixed-temperature grid complete: Teacher seed=$TEACHER_SEED"
