#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
RSEEDS=(41 42); SSEEDS=(42 43 44)
EARLY_ROOT="$Main_Data_Path/class_in_class/imagenette_early_teacher_downstream"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_alpha_intervention"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_alpha_intervention}"
VAL_DIR="$val_dir/imagenet-nette/test"
mkdir -p "$EXP_ROOT/fkd" "$EXP_ROOT/per_class" "$EXP_ROOT/post_eval" "$EXP_ROOT/analysis" "$LOG_ROOT"
fail(){ echo "Alpha intervention failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }
alpha_tag(){ printf '%.3f' "$1" | tr '.' 'p'; }

transform_one(){
    local family="$1" recovery="$2"
    local input alphas
    if [[ "$family" == c1_e300 ]]; then
        input="$EARLY_ROOT/tseed43/fkd/real__c1_e300_e299_ref_rseed${recovery}_bs10_ipc10"
        alphas="0.70 0.85 1.00 1.075 1.20 1.50 1.80"
    else
        input="$EARLY_ROOT/tseed43/fkd/real__c100_e100_e099_ref_rseed${recovery}_bs10_ipc10"
        alphas="0.70 0.85 0.93 1.00 1.20 1.50 1.80"
    fi
    [[ -d "$input" ]] || { echo "missing input FKD: $input" >&2; return 1; }
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/class_in_class/transform_fkd_within_class_alpha.py" \
        --input-root "$input" --output-base "$EXP_ROOT/fkd/${family}_rseed${recovery}" \
        --alphas $alphas --source-temperature 20 --classes 10 --output-dtype fp16 \
        > "$LOG_ROOT/transform_${family}_r${recovery}.log" 2>&1
}

echo "[1/4] Two-pass global normalization and alpha FKD transforms"
pids=()
for recovery in "${RSEEDS[@]}"; do
    transform_one c1_e300 "$recovery" & pids+=("$!")
    transform_one c100_e100 "$recovery" & pids+=("$!")
    if (( ${#pids[@]} == 2 )); then
        wait_jobs "${pids[@]}" || fail transform
        pids=()
    fi
done

echo "[2/4] Preflight alpha=1 replay and R'=alpha^2 R identity"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/audit_imagenette_alpha_transform_preflight.py" \
    --fkd-root "$EXP_ROOT/fkd" > "$LOG_ROOT/transform_preflight.log" 2>&1 \
    || fail transform_preflight

post_one(){
    local family="$1" alpha="$2" recovery="$3" student="$4" gpu="$5"
    local tag fkd source result valid
    tag="$(alpha_tag "$alpha")"
    fkd="$EXP_ROOT/fkd/${family}_rseed${recovery}/alpha_${tag}"
    source="$FACTORIAL_ROOT/real_sets/tseed43_rseed${recovery}"
    result="$EXP_ROOT/per_class/${family}_alpha${tag}_rseed${recovery}_sseed${student}.json"
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and q.get('training_target')=='fkd_soft_label')")"
        [[ "$valid" == True ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    [[ "$(find "$fkd" -type f -name 'batch_*.tar' | wc -l)" == 3000 ]] || return 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" \
        --fkd-path "$fkd" --mix-type cutmix --temperature 1 \
        --model ResNet18 --ipc 10 \
        --exp-name "alpha_${family}_a${tag}_r${recovery}_s${student}" \
        --original-data-path "$source" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 10 --epochs 300 --dataset-name imagenet-nette \
        --gradient-accumulation-steps 2 --cos --workers "$WORKERS" \
        --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005 \
        --eta-override 1 --train-seed "$student" --persistent-workers \
        --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" \
        > "$LOG_ROOT/post_${family}_a${tag}_r${recovery}_s${student}.log" 2>&1
}

echo "[3/4] Post-eval alpha arms, two processes per A100"
pids=(); task=0
for family in c1_e300 c100_e100; do
    if [[ "$family" == c1_e300 ]]; then
        alphas=(0.70 0.85 1.00 1.075 1.20 1.50 1.80)
    else
        alphas=(0.70 0.85 0.93 1.00 1.20 1.50 1.80)
    fi
    for alpha in "${alphas[@]}"; do
        for recovery in "${RSEEDS[@]}"; do
            for student in "${SSEEDS[@]}"; do
                gpu="$GPU0"; (( task % 2 )) && gpu="$GPU1"; task=$((task + 1))
                post_one "$family" "$alpha" "$recovery" "$student" "$gpu" & pids+=("$!")
                if (( ${#pids[@]} == 4 )); then
                    wait_jobs "${pids[@]}" || fail post_eval
                    pids=()
                fi
            done
        done
    done
done
(( ${#pids[@]} == 0 )) || wait_jobs "${pids[@]}" || fail post_eval

echo "[4/4] Summarize alpha curves and alpha=1 replay"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -u "$ROOT/class_in_class/summarize_imagenette_alpha_intervention.py" \
    --experiment-root "$EXP_ROOT" --existing-root "$EARLY_ROOT" \
    --output "$EXP_ROOT/analysis/alpha_intervention_summary.json" \
    > "$LOG_ROOT/summarize.log" 2>&1

echo "Complete: $EXP_ROOT/analysis/alpha_intervention_summary.json"
