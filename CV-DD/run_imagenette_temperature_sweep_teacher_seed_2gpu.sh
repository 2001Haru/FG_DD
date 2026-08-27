#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED to 43 or 44}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
STAGE="${STAGE:-all}"
[[ "$STAGE" == relabel || "$STAGE" == post || "$STAGE" == all ]] || {
    echo "STAGE must be relabel, post, or all" >&2
    exit 1
}

TEMPERATURES=(1 2 4 8 20 46 100 200 400 800 1600)
RSEEDS=(41 42)
SSEEDS=(42 43)
ROWS=(real c1)
COLS=(c1 random100)

RANDOM_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
MATCH_ROOT="$Main_Data_Path/class_in_class/imagenette_entropy_matched"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_temperature_sweep_ipc10"
TEACHER_ROOT="$RANDOM_ROOT/tseed${TEACHER_SEED}"
TEACHER_C1="$TEACHER_ROOT/models/random_c1_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
TEACHER_R100="$TEACHER_ROOT/models/random_c100_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
MAP_C1="$TEACHER_ROOT/data/random_c1_pseed42/hierarchy.json"
MAP_R100="$TEACHER_ROOT/data/random_c100_pseed42/hierarchy.json"
FKD_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/fkd"
RESULT_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/per_class"
POST_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/post_eval"
VAL_DIR="$val_dir/imagenet-nette/test"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_temperature_sweep_ipc10/tseed${TEACHER_SEED}}"
mkdir -p "$FKD_ROOT" "$RESULT_ROOT" "$POST_ROOT" "$LOG_ROOT"

fail() { echo "ImageNette temperature sweep failed: $*" >&2; exit 1; }
wait_jobs() {
    local status=0 pid
    for pid in "$@"; do wait "$pid" || status=1; done
    return "$status"
}
tag_for() { local col="$1" temp="$2"; echo "${col}_T${temp/./p}"; }
source_for() {
    local row="$1" rseed="$2"
    if [[ "$row" == real ]]; then
        echo "$FACTORIAL_ROOT/real_sets/tseed${TEACHER_SEED}_rseed${rseed}"
    else
        echo "$RANDOM_ROOT/tseed${TEACHER_SEED}/synthetic/cic_t_c1_ipc10_rseed${rseed}"
    fi
}
teacher_for() { [[ "$1" == c1 ]] && echo "$TEACHER_C1" || echo "$TEACHER_R100"; }
mapping_for() { [[ "$1" == c1 ]] && echo "$MAP_C1" || echo "$MAP_R100"; }
heads_for() { [[ "$1" == c1 ]] && echo 10 || echo 1000; }
is_reused() {
    local col="$1" temp="$2"
    [[ "$temp" == 20 || ( "$col" == c1 && "$temp" == 46 ) ]]
}
reference_result() {
    local row="$1" col="$2" temp="$3" rseed="$4" sseed="$5"
    if [[ "$temp" == 20 ]]; then
        if [[ "$row" == real ]]; then
            echo "$FACTORIAL_ROOT/tseed${TEACHER_SEED}/per_class/real__${col}_rseed${rseed}_sseed${sseed}.json"
        elif [[ "$col" == c1 ]]; then
            echo "$RANDOM_ROOT/tseed${TEACHER_SEED}/per_class/c1_rseed${rseed}_sseed${sseed}.json"
        else
            echo "$FACTORIAL_ROOT/tseed${TEACHER_SEED}/per_class/c1__random100_rseed${rseed}_sseed${sseed}.json"
        fi
    elif [[ "$col" == c1 && "$temp" == 46 ]]; then
        echo "$MATCH_ROOT/tseed${TEACHER_SEED}/per_class/ipc10_${row}__c1_T46p0_rseed${rseed}_sseed${sseed}.json"
    else
        return 1
    fi
}

for row in "${ROWS[@]}"; do
    for rseed in "${RSEEDS[@]}"; do
        [[ -d "$(source_for "$row" "$rseed")" ]] || fail "missing source row=$row recovery=$rseed"
    done
done
[[ -f "$TEACHER_C1" && -f "$TEACHER_R100" ]] || fail "missing Teacher checkpoint"
[[ -f "$MAP_C1" && -f "$MAP_R100" ]] || fail "missing hierarchy mapping"
[[ "$(find "$VAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)" == 3925 ]] \
    || fail "validation split is not the full 3925-image test set"

# T20 and C1@T46 were already evaluated with the same sources, views, and seeds.
for temp in "${TEMPERATURES[@]}"; do
    for row in "${ROWS[@]}"; do
        for col in "${COLS[@]}"; do
            if is_reused "$col" "$temp"; then
                for rseed in "${RSEEDS[@]}"; do
                    for sseed in "${SSEEDS[@]}"; do
                        ref="$(reference_result "$row" "$col" "$temp" "$rseed" "$sseed")"
                        [[ -f "$ref" ]] || fail "missing reusable result: $ref"
                    done
                done
            fi
        done
    done
done

relabel_one() {
    local row="$1" col="$2" temp="$3" rseed="$4" gpu="$5"
    local src teacher mapping heads tag base final count
    is_reused "$col" "$temp" && return 0
    src="$(source_for "$row" "$rseed")"
    teacher="$(teacher_for "$col")"
    mapping="$(mapping_for "$col")"
    heads="$(heads_for "$col")"
    tag="$(tag_for "$col" "$temp")"
    base="$FKD_ROOT/${row}__${tag}_rseed${rseed}"
    final="${base}_bs10_ipc10"
    count=0
    [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return 0
    (( count > 0 )) && mv "$final" "${final}.partial_$(date +%Y%m%d_%H%M%S)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$src" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" \
        --teacher-model-name ResNet18 --teacher-num-classes "$heads" \
        --teacher-mapping "$mapping" --marginalize-temperature "$temp" \
        --gpu 0 --batch-size 10 --workers "$WORKERS" --persistent-workers \
        --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 \
        --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix \
        > "$LOG_ROOT/relabel_${row}__${tag}_r${rseed}.log" 2>&1
    count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    [[ "$count" == 3000 ]]
}

post_one() {
    local row="$1" col="$2" temp="$3" rseed="$4" sseed="$5" gpu="$6"
    local src tag fkd result
    is_reused "$col" "$temp" && return 0
    src="$(source_for "$row" "$rseed")"
    tag="$(tag_for "$col" "$temp")"
    fkd="$FKD_ROOT/${row}__${tag}_rseed${rseed}_bs10_ipc10"
    result="$RESULT_ROOT/${row}__${tag}_rseed${rseed}_sseed${sseed}.json"
    [[ -f "$result" ]] && return 0
    [[ "$(find "$fkd" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] \
        || return 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -u "$ROOT/validate/train_fkd.py" \
        --fkd-path "$fkd" --mix-type cutmix --temperature "$temp" \
        --model ResNet18 --ipc 10 \
        --exp-name "temp_sweep_${row}__${tag}_t${TEACHER_SEED}_r${rseed}_s${sseed}" \
        --original-data-path "$src" --output-dir "$POST_ROOT" --batch-size 10 \
        --epochs 300 --dataset-name imagenet-nette --gradient-accumulation-steps 2 \
        --cos --workers "$WORKERS" --fkd_seed 42 --adamw-weight-decay 0.01 \
        --adamw-lr-override 0.0005 --eta-override 1 --train-seed "$sseed" \
        --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" \
        > "$LOG_ROOT/post_${row}__${tag}_r${rseed}_s${sseed}.log" 2>&1
}

if [[ "$STAGE" == relabel || "$STAGE" == all ]]; then
    echo "[Relabel] temperatures=${TEMPERATURES[*]} teacher_seed=$TEACHER_SEED"
    pids=(); task=0
    for temp in "${TEMPERATURES[@]}"; do
        for rseed in "${RSEEDS[@]}"; do
            for row in "${ROWS[@]}"; do
                for col in "${COLS[@]}"; do
                    is_reused "$col" "$temp" && continue
                    gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
                    relabel_one "$row" "$col" "$temp" "$rseed" "$gpu" & pids+=("$!")
                    if (( ${#pids[@]} == 2 )); then
                        wait_jobs "${pids[@]}" || fail relabel
                        pids=()
                    fi
                done
            done
        done
    done
    (( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail relabel
fi

if [[ "$STAGE" == post || "$STAGE" == all ]]; then
    echo "[Post-eval] 2 recovery x 2 student seeds"
    pids=(); task=0
    for temp in "${TEMPERATURES[@]}"; do
        for rseed in "${RSEEDS[@]}"; do
            for row in "${ROWS[@]}"; do
                for col in "${COLS[@]}"; do
                    is_reused "$col" "$temp" && continue
                    for sseed in "${SSEEDS[@]}"; do
                        gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
                        post_one "$row" "$col" "$temp" "$rseed" "$sseed" "$gpu" & pids+=("$!")
                        if (( ${#pids[@]} == 4 )); then
                            wait_jobs "${pids[@]}" || fail post
                            pids=()
                        fi
                    done
                done
            done
        done
    done
    (( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post
fi

echo "Temperature sweep complete: teacher_seed=$TEACHER_SEED stage=$STAGE"
