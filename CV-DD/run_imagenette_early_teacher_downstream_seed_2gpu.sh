#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 43 or 44}"
[[ "$TEACHER_SEED" == 43 || "$TEACHER_SEED" == 44 ]] || exit 1
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RSEEDS=(41 42); SSEEDS=(42 43)
TRAJECTORY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_trajectories"
EXISTING_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
SWEEP_ROOT="$Main_Data_Path/class_in_class/imagenette_temperature_sweep_ipc10"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_downstream"
SEED_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}"
FKD_ROOT="$SEED_ROOT/fkd"; RESULT_ROOT="$SEED_ROOT/per_class"; POST_ROOT="$SEED_ROOT/post_eval"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_early_teacher_downstream/tseed${TEACHER_SEED}}"
VAL_DIR="$val_dir/imagenet-nette/test"
mkdir -p "$FKD_ROOT" "$RESULT_ROOT" "$POST_ROOT" "$LOG_ROOT"
fail(){ echo "Early Teacher downstream failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }

echo "[1/4] Select checkpoints and audit endpoint reuse"
python -u "$ROOT/class_in_class/select_imagenette_early_teacher_checkpoints.py" \
    --trajectory-root "$TRAJECTORY_ROOT" --existing-teacher-root "$EXISTING_ROOT" \
    --teacher-seed "$TEACHER_SEED" --output-root "$EXP_ROOT" \
    > "$LOG_ROOT/selection.log" 2>&1 || fail selection
PLAN="$SEED_ROOT/selection_early.tsv"
[[ "$(wc -l < "$PLAN")" == 20 ]] \
    || fail "selection plan must contain ten fixed epochs for both C1 and C100"

source_for(){
    local source="$1" rseed="$2"
    if [[ "$source" == real ]]; then
        echo "$FACTORIAL_ROOT/real_sets/tseed${TEACHER_SEED}_rseed${rseed}"
    else
        echo "$EXISTING_ROOT/tseed${TEACHER_SEED}/synthetic/cic_t_c1_ipc10_rseed${rseed}"
    fi
}
mapping_for(){
    local c="$1"
    echo "$TRAJECTORY_ROOT/tseed${TEACHER_SEED}/data/random_c${c}_pseed42/hierarchy.json"
}
heads_for(){ local c="$1"; echo $((10 * c)); }
tag_for(){
    local c="$1" label="$2" epoch="$3" mode="$4"
    echo "c${c}_${label}_e$(printf '%03d' "$epoch")_${mode}"
}

for source in real c1; do for rseed in "${RSEEDS[@]}"; do
    directory="$(source_for "$source" "$rseed")"
    [[ -d "$directory" ]] || fail "missing source=$source r=$rseed"
    [[ "$(find "$directory" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 100 ]] \
        || fail "source must contain IPC10=100 images: $directory"
done; done
[[ "$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 3925 ]] \
    || fail "invalid test split"

relabel_one(){
    local c="$1" label="$2" epoch="$3" predicted="$4" teacher_view="$5"
    local source="$6" mode="$7" rseed="$8" gpu="$9"
    local temperature=20 tag mapping heads data base final count
    [[ "$mode" == pred ]] && temperature="$predicted"
    tag="$(tag_for "$c" "$label" "$epoch" "$mode")"
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

echo "[2/4] Relabel C1/C100 at ten fixed epochs x two temperatures x two sources"
pids=(); task=0
while IFS=$'\t' read -r c label epoch actual_val sd_z predicted teacher_view; do
    for source in real c1; do for mode in ref pred; do for rseed in "${RSEEDS[@]}"; do
        [[ "$label" == e004 && "$mode" == pred ]] && continue
        gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
        relabel_one "$c" "$label" "$epoch" "$predicted" "$teacher_view" "$source" "$mode" "$rseed" "$gpu" & pids+=("$!")
        if (( ${#pids[@]} == 2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi
    done; done; done
done < "$PLAN"
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail relabel

post_one(){
    local c="$1" label="$2" epoch="$3" predicted="$4" source="$5" mode="$6"
    local rseed="$7" sseed="$8" gpu="$9"
    local temperature=20 tag data fkd result
    [[ "$mode" == pred ]] && temperature="$predicted"
    tag="$(tag_for "$c" "$label" "$epoch" "$mode")"; data="$(source_for "$source" "$rseed")"
    fkd="$FKD_ROOT/${source}__${tag}_rseed${rseed}_bs10_ipc10"
    result="$RESULT_ROOT/${source}__${tag}_rseed${rseed}_sseed${sseed}.json"
    [[ -f "$result" ]] && return
    [[ "$(find "$fkd" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] || return 1
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

echo "[3/4] Post-eval with two processes per A100"
pids=(); task=0
while IFS=$'\t' read -r c label epoch actual_val sd_z predicted teacher_view; do
    for source in real c1; do for mode in ref pred; do for rseed in "${RSEEDS[@]}"; do for sseed in "${SSEEDS[@]}"; do
        [[ "$label" == e004 && "$mode" == pred ]] && continue
        gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
        post_one "$c" "$label" "$epoch" "$predicted" "$source" "$mode" "$rseed" "$sseed" "$gpu" & pids+=("$!")
        if (( ${#pids[@]} == 4 )); then wait_jobs "${pids[@]}" || fail post; pids=(); fi
    done; done; done; done
done < "$PLAN"
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post

echo "[4/4] Complete Teacher seed=$TEACHER_SEED; trajectory final endpoints were evaluated directly"
