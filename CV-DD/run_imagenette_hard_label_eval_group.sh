#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
C_VALUES_TEXT="${C_VALUES:?set C_VALUES, for example: C_VALUES='1 2 5 10 20'}"
read -r -a C_VALUES_ARRAY <<< "$C_VALUES_TEXT"
RSEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"
read -r -a RSEEDS <<< "$RSEEDS_TEXT"
read -r -a SSEEDS <<< "$SSEEDS_TEXT"
MASTER_ROOT="${MASTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
HARD_PROTOCOL="${HARD_PROTOCOL:-ImageNette IPC10 ResNet18; existing CiC-T synthetic images; hard coarse10 labels; no FKD/relabel; 300 epochs BS10 AdamW LR5e-4 eta1}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_hard_label_eval}"
mkdir -p "$LOG_ROOT"

fail(){ echo "ImageNette hard-label evaluation failed: $*" >&2; exit 1; }

VAL_IMAGE_COUNT="$(find "$VAL_DIR" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)"
VAL_CLASS_COUNT="$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[[ "$VAL_IMAGE_COUNT" == 3925 && "$VAL_CLASS_COUNT" == 10 ]] \
    || fail "invalid ImageNette test set: path=$VAL_DIR images=$VAL_IMAGE_COUNT classes=$VAL_CLASS_COUNT"

validate_assets(){
    local teacher_seed="$1" c rseed syn count
    for c in "${C_VALUES_ARRAY[@]}"; do
        for rseed in "${RSEEDS[@]}"; do
            syn="$MASTER_ROOT/tseed${teacher_seed}/synthetic/cic_t_c${c}_ipc10_rseed${rseed}"
            [[ -d "$syn" ]] || fail "missing synthetic set: $syn"
            count="$(find "$syn" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
            [[ "$count" == 100 ]] || fail "synthetic set has $count images, expected 100: $syn"
            classes="$(find "$syn" -mindepth 1 -maxdepth 1 -type d | wc -l)"
            [[ "$classes" == 10 ]] || fail "synthetic set has $classes coarse class dirs, expected 10: $syn"
        done
    done
}

run_teacher_stream(){
    local teacher_seed="$1" gpu="$2"
    local teacher_root="$MASTER_ROOT/tseed${teacher_seed}"
    local result_root="$teacher_root/hard_per_class"
    local post_root="$teacher_root/hard_post_eval"
    local stream_logs="$LOG_ROOT/tseed${teacher_seed}"
    local c rseed sseed syn result log result_valid
    mkdir -p "$result_root" "$post_root" "$stream_logs"

    for sseed in "${SSEEDS[@]}"; do
        for rseed in "${RSEEDS[@]}"; do
            for c in "${C_VALUES_ARRAY[@]}"; do
                syn="$teacher_root/synthetic/cic_t_c${c}_ipc10_rseed${rseed}"
                result="$result_root/c${c}_rseed${rseed}_sseed${sseed}.json"
                log="$stream_logs/c${c}_rseed${rseed}_sseed${sseed}.log"

                if [[ -f "$result" ]]; then
                    result_valid="$(python -c "import json,os; q=json.load(open('$result')); print(q.get('training_target')=='hard_coarse_label' and int(q.get('validation_images',-1))==3925 and os.path.realpath(q.get('validation_dir',''))==os.path.realpath('$VAL_DIR'))")"
                    [[ "$result_valid" == "True" ]] && continue
                    mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
                fi

                CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
                python -u "$ROOT/validate/train_fkd.py" \
                    --hard-label --model ResNet18 --ipc 10 \
                    --exp-name "imagenette_hard_c${c}_tseed${teacher_seed}_rseed${rseed}_sseed${sseed}" \
                    --original-data-path "$syn" --output-dir "$post_root" \
                    --batch-size 10 --epochs 300 --dataset-name imagenet-nette \
                    --gradient-accumulation-steps 2 --cos --workers "$WORKERS" \
                    --fkd_seed 42 --adamw-weight-decay 0.01 \
                    --adamw-lr-override 0.0005 --eta-override 1 \
                    --train-seed "$sseed" --persistent-workers \
                    --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
                    > "$log" 2>&1 || {
                        echo "failed: tseed=$teacher_seed C=$c rseed=$rseed sseed=$sseed; log=$log" >&2
                        return 1
                    }
                [[ -f "$result" ]] || {
                    echo "missing result after evaluation: $result" >&2; return 1;
                }
            done
        done
    done
}

validate_assets 43
validate_assets 44
echo "Hard-label streams: Teacher seed43 on GPU$GPU0; Teacher seed44 on GPU$GPU1; C=${C_VALUES_ARRAY[*]}"
run_teacher_stream 43 "$GPU0" & pid43=$!
run_teacher_stream 44 "$GPU1" & pid44=$!
status=0
wait "$pid43" || status=1
wait "$pid44" || status=1
(( status == 0 )) || fail "one or both Teacher streams"

c_tag="${C_VALUES_ARRAY[*]}"; c_tag="${c_tag// /_}"
read -r -a RSEED_ARRAY <<< "$RSEEDS_TEXT"
read -r -a SSEED_ARRAY <<< "$SSEEDS_TEXT"
summary="$MASTER_ROOT/analysis/hard_label_summary_c${c_tag}.json"
python "$ROOT/class_in_class/summarize_imagenette_cic_t_teacher_seeds.py" \
    --master-root "$MASTER_ROOT" --teacher-seeds 43 44 \
    --recovery-seeds "${RSEED_ARRAY[@]}" --student-seeds "${SSEED_ARRAY[@]}" \
    --c-values "${C_VALUES_ARRAY[@]}" --per-class-subdir hard_per_class \
    --protocol "$HARD_PROTOCOL" \
    --output "$summary"
echo "Complete: $summary"
