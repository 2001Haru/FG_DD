#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/config.sh"

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
ROWS_TEXT="${ROWS:?set ROWS}"; read -r -a ROWS_ARRAY <<< "$ROWS_TEXT"
RSEEDS_TEXT="${RECOVERY_SEEDS:-41 42 43}"; read -r -a RSEEDS <<< "$RSEEDS_TEXT"
SSEEDS_TEXT="${STUDENT_SEEDS:-42 43 44}"; read -r -a SSEEDS <<< "$SSEEDS_TEXT"
RANDOM_ROOT="${RANDOM_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44}"
CLUSTER_ROOT="${CLUSTER_ROOT:-$Main_Data_Path/class_in_class/imagenette_cic_t_dinov2_cluster_seed42}"
MATRIX_ROOT="${MATRIX_ROOT:-$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100}"
VAL_DIR="${VAL_DIR:-$val_dir/imagenet-nette/test}"
REAL_TRAIN="${REAL_TRAIN:-$val_dir/imagenet-nette/train}"
VIEW_SEED="${VIEW_SEED:-42}"; TEMPERATURE="${TEMPERATURE:-20}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_labeler_factorial_c100}"
mkdir -p "$MATRIX_ROOT" "$LOG_ROOT"

fail(){ echo "ImageNette labeler factorial failed: $*" >&2; exit 1; }
wait_pair(){ local a="$1" b="$2" status=0; wait "$a" || status=1; wait "$b" || status=1; return "$status"; }

columns_for_row(){
    case "$1" in
        real) printf '%s\n' "hard c1 random100 cluster100" ;;
        c1) printf '%s\n' "random100 cluster100" ;;
        random100) printf '%s\n' "c1 cluster100" ;;
        cluster100) printf '%s\n' "c1 random100" ;;
        *) return 1 ;;
    esac
}

source_for(){
    local row="$1" teacher_seed="$2" recovery_seed="$3"
    case "$row" in
        real) printf '%s\n' "$MATRIX_ROOT/real_sets/tseed${teacher_seed}_rseed${recovery_seed}" ;;
        c1) printf '%s\n' "$RANDOM_ROOT/tseed${teacher_seed}/synthetic/cic_t_c1_ipc10_rseed${recovery_seed}" ;;
        random100) printf '%s\n' "$RANDOM_ROOT/tseed${teacher_seed}/synthetic/cic_t_c100_ipc10_rseed${recovery_seed}" ;;
        cluster100) printf '%s\n' "$CLUSTER_ROOT/tseed${teacher_seed}/synthetic/cic_t_c100_ipc10_rseed${recovery_seed}" ;;
        *) return 1 ;;
    esac
}

teacher_for(){
    local column="$1" teacher_seed="$2"
    case "$column" in
        c1) printf '%s\n' "$RANDOM_ROOT/tseed${teacher_seed}/models/random_c1_pseed42_tseed${teacher_seed}/ResNet18.pth" ;;
        random100) printf '%s\n' "$RANDOM_ROOT/tseed${teacher_seed}/models/random_c100_pseed42_tseed${teacher_seed}/ResNet18.pth" ;;
        cluster100) printf '%s\n' "$CLUSTER_ROOT/tseed${teacher_seed}/models/dinov2_cluster_c100_seed42_tseed${teacher_seed}/ResNet18.pth" ;;
        *) return 1 ;;
    esac
}

mapping_for(){
    local column="$1" teacher_seed="$2"
    case "$column" in
        c1) printf '%s\n' "$RANDOM_ROOT/tseed${teacher_seed}/data/random_c1_pseed42/hierarchy.json" ;;
        random100) printf '%s\n' "$RANDOM_ROOT/tseed${teacher_seed}/data/random_c100_pseed42/hierarchy.json" ;;
        cluster100) printf '%s\n' "$CLUSTER_ROOT/data/dinov2_cluster_c100_seed42/hierarchy.json" ;;
        *) return 1 ;;
    esac
}

heads_for(){ case "$1" in c1) echo 10 ;; random100|cluster100) echo 1000 ;; *) return 1 ;; esac; }

echo "[1/3] Preflight and random-real subsets"
[[ -d "$VAL_DIR" && -d "$REAL_TRAIN" ]] || fail "missing official ImageNette train/test"
for teacher_seed in 43 44; do
    for recovery_seed in "${RSEEDS[@]}"; do
        if [[ " ${ROWS_ARRAY[*]} " == *" real "* ]]; then
            subset="$MATRIX_ROOT/real_sets/tseed${teacher_seed}_rseed${recovery_seed}"
            subset_seed=$((42000000 + teacher_seed * 100003 + recovery_seed * 1009))
            python "$ROOT/class_in_class/prepare_imagenette_random_real_ipc10.py" \
                --source-train "$REAL_TRAIN" --output-dir "$subset" --seed "$subset_seed" \
                --images-per-class 10 --repair-invalid-output \
                > "$LOG_ROOT/prepare_real_t${teacher_seed}_r${recovery_seed}.log" 2>&1
        fi
        for row in "${ROWS_ARRAY[@]}"; do
            source="$(source_for "$row" "$teacher_seed" "$recovery_seed")"
            [[ -d "$source" ]] || fail "missing source: row=$row tseed=$teacher_seed rseed=$recovery_seed"
            images="$(find "$source" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
            classes="$(find "$source" -mindepth 1 -maxdepth 1 -type d | wc -l)"
            [[ "$images" == 100 && "$classes" == 10 ]] \
                || fail "invalid source row=$row tseed=$teacher_seed rseed=$recovery_seed images=$images classes=$classes"
        done
    done
    for column in c1 random100 cluster100; do
        teacher="$(teacher_for "$column" "$teacher_seed")"
        mapping="$(mapping_for "$column" "$teacher_seed")"
        [[ -f "$teacher" && -f "$mapping" ]] \
            || fail "missing labeler assets: column=$column tseed=$teacher_seed"
    done
done

relabel_one(){
    local row="$1" column="$2" teacher_seed="$3" recovery_seed="$4" gpu="$5"
    local source teacher mapping heads base final count archive log
    source="$(source_for "$row" "$teacher_seed" "$recovery_seed")"
    teacher="$(teacher_for "$column" "$teacher_seed")"
    mapping="$(mapping_for "$column" "$teacher_seed")"
    heads="$(heads_for "$column")"
    base="$MATRIX_ROOT/tseed${teacher_seed}/fkd/${row}__${column}_rseed${recovery_seed}"
    final="${base}_bs10_ipc10"
    log="$LOG_ROOT/tseed${teacher_seed}/relabel_${row}__${column}_r${recovery_seed}.log"
    mkdir -p "$(dirname "$log")" "$(dirname "$base")"
    # Recover outputs created by the historical .jpg-only counter (real .JPEG
    # subsets were incorrectly named ipc0). IPC only affected the directory
    # suffix, so a complete 3000-tar tree can be atomically renamed and reused.
    shopt -s nullglob
    for candidate in "${base}_bs10_ipc"*; do
        ipc_suffix="${candidate#${base}_bs10_ipc}"
        [[ "$ipc_suffix" =~ ^[0-9]+$ ]] || continue
        if [[ "$candidate" != "$final" && -d "$candidate" ]]; then
            candidate_count="$(find "$candidate" -type f -name 'batch_*.tar' | wc -l)"
            if [[ "$ipc_suffix" == 0 && "$candidate_count" == 3000 && ! -e "$final" ]]; then
                mv "$candidate" "$final"
                echo "Recovered complete mislabeled FKD tree: $candidate -> $final"
                continue
            fi
            stale_archive="${candidate}.invalid_image_count_$(date +%Y%m%d_%H%M%S)"
            [[ ! -e "$stale_archive" ]] || { echo "stale archive exists: $stale_archive" >&2; shopt -u nullglob; return 1; }
            mv "$candidate" "$stale_archive"
            echo "Archived stale FKD directory: $candidate -> $stale_archive"
        fi
    done
    shopt -u nullglob
    count=0; [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) && return
    if (( count > 0 )); then
        archive="${final}.invalid_$(date +%Y%m%d_%H%M%S)"
        mv "$final" "$archive"
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/relabel/relabel.py" \
        --syn-data-path "$source" --fkd-path "$base" \
        --model-pool-dir "$(dirname "$teacher")" --teacher-model-name ResNet18 \
        --teacher-num-classes "$heads" --teacher-mapping "$mapping" \
        --marginalize-temperature "$TEMPERATURE" --gpu 0 --batch-size 10 --workers "$WORKERS" \
        --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette \
        --epochs 300 --fkd-seed "$VIEW_SEED" --seed "$VIEW_SEED" \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 \
        --mode fkd_save --mix-type cutmix > "$log" 2>&1; then
        echo "Relabel failed: row=$row column=$column tseed=$teacher_seed rseed=$recovery_seed; log=$log" >&2
        return 1
    fi
    count=0; [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar' | wc -l)"
    (( count == 3000 )) || { echo "incomplete FKD: $final ($count/3000)" >&2; return 1; }
}

post_one(){
    local row="$1" column="$2" teacher_seed="$3" recovery_seed="$4" student_seed="$5" gpu="$6"
    local source result post_root log fkd args=()
    source="$(source_for "$row" "$teacher_seed" "$recovery_seed")"
    result="$MATRIX_ROOT/tseed${teacher_seed}/per_class/${row}__${column}_rseed${recovery_seed}_sseed${student_seed}.json"
    post_root="$MATRIX_ROOT/tseed${teacher_seed}/post_eval"
    log="$LOG_ROOT/tseed${teacher_seed}/validate_${row}__${column}_r${recovery_seed}_s${student_seed}.log"
    mkdir -p "$(dirname "$result")" "$post_root" "$(dirname "$log")"
    if [[ -f "$result" ]]; then
        valid="$(python -c "import json,os; q=json.load(open('$result')); print(int(q.get('validation_images',-1))==3925 and os.path.realpath(q.get('validation_dir',''))==os.path.realpath('$VAL_DIR'))")"
        [[ "$valid" == "True" ]] && return
        mv "$result" "${result}.invalid_$(date +%Y%m%d_%H%M%S)"
    fi
    if [[ "$column" == hard ]]; then
        args+=(--hard-label)
    else
        fkd="$MATRIX_ROOT/tseed${teacher_seed}/fkd/${row}__${column}_rseed${recovery_seed}_bs10_ipc10"
        [[ -d "$fkd" ]] || { echo "missing FKD: $fkd" >&2; return 1; }
        args+=(--fkd-path "$fkd" --mix-type cutmix --temperature "$TEMPERATURE")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "$ROOT/validate/train_fkd.py" "${args[@]}" \
        --model ResNet18 --ipc 10 \
        --exp-name "factorial_${row}__${column}_t${teacher_seed}_r${recovery_seed}_s${student_seed}" \
        --original-data-path "$source" --output-dir "$post_root" --batch-size 10 \
        --epochs 300 --dataset-name imagenet-nette --gradient-accumulation-steps 2 \
        --cos --workers "$WORKERS" --fkd_seed "$VIEW_SEED" --adamw-weight-decay 0.01 \
        --adamw-lr-override 0.0005 --eta-override 1 --train-seed "$student_seed" \
        --persistent-workers --val-dir "$VAL_DIR" --disable-wandb \
        --per-class-output "$result" > "$log" 2>&1
    [[ -f "$result" ]] || { echo "missing result: $result" >&2; return 1; }
}

stream(){
    local teacher_seed="$1" gpu="$2" row column recovery_seed student_seed columns
    for row in "${ROWS_ARRAY[@]}"; do
        columns="$(columns_for_row "$row")"
        for recovery_seed in "${RSEEDS[@]}"; do
            for column in $columns; do
                if [[ "$column" != hard ]]; then
                    echo "[Relabel] t=$teacher_seed r=$recovery_seed row=$row col=$column"
                    relabel_one "$row" "$column" "$teacher_seed" "$recovery_seed" "$gpu" || return 1
                fi
                for student_seed in "${SSEEDS[@]}"; do
                    echo "[Post] t=$teacher_seed r=$recovery_seed s=$student_seed row=$row col=$column"
                    post_one "$row" "$column" "$teacher_seed" "$recovery_seed" "$student_seed" "$gpu" || return 1
                done
            done
        done
    done
}

echo "[2/3] Missing cells: rows=${ROWS_ARRAY[*]}; tseed43 GPU$GPU0; tseed44 GPU$GPU1"
stream 43 "$GPU0" & pid43=$!
stream 44 "$GPU1" & pid44=$!
wait_pair "$pid43" "$pid44" || fail "relabel/post-eval stream"
echo "[3/3] Partial group complete: rows=${ROWS_ARRAY[*]}"
