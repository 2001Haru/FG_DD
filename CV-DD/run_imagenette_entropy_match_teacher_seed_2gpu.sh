#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$ROOT/config.sh"
TEACHER_SEED="${TEACHER_SEED:?set TEACHER_SEED}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; WORKERS="${WORKERS:-8}"
STAGE="${STAGE:-relabel}"; [[ "$STAGE" == relabel || "$STAGE" == post || "$STAGE" == all ]] || { echo "STAGE must be relabel/post/all" >&2; exit 1; }
C1_MATCH_T="${C1_MATCH_T:-43.4}"; R100_MATCH_T="${R100_MATCH_T:-9.2}"
RSEEDS=(41 42 43); SSEEDS=(42 43 44); IPCS=(1 10)
RANDOM_ROOT="$Main_Data_Path/class_in_class/imagenette_cic_t_official_split_lr0p1_tseeds43_44"
FACTORIAL_ROOT="$Main_Data_Path/class_in_class/imagenette_labeler_factorial_c100"
IPC_ROOT="$Main_Data_Path/class_in_class/imagenette_ipc1_50_main_table"
EXP_ROOT="$Main_Data_Path/class_in_class/imagenette_entropy_matched"
TEACHER_ROOT="$RANDOM_ROOT/tseed${TEACHER_SEED}"
TEACHER_C1="$TEACHER_ROOT/models/random_c1_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
TEACHER_R100="$TEACHER_ROOT/models/random_c100_pseed42_tseed${TEACHER_SEED}/ResNet18.pth"
MAP_C1="$TEACHER_ROOT/data/random_c1_pseed42/hierarchy.json"; MAP_R100="$TEACHER_ROOT/data/random_c100_pseed42/hierarchy.json"
FKD_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/fkd"; RESULT_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/per_class"; POST_ROOT="$EXP_ROOT/tseed${TEACHER_SEED}/post_eval"
VAL_DIR="$val_dir/imagenet-nette/test"; LOG_ROOT="${LOG_ROOT:-$ROOT/logs/imagenette_entropy_matched/tseed${TEACHER_SEED}}"
mkdir -p "$FKD_ROOT" "$RESULT_ROOT" "$POST_ROOT" "$LOG_ROOT"
fail(){ echo "Entropy-match pipeline failed: $*" >&2; exit 1; }
wait_jobs(){ local status=0 pid; for pid in "$@"; do wait "$pid" || status=1; done; return "$status"; }
source_for(){ local ipc="$1" row="$2" r="$3"; if [[ "$ipc" == 1 ]]; then echo "$IPC_ROOT/tseed${TEACHER_SEED}/sources/${row}_ipc1_rseed${r}"; elif [[ "$row" == real ]]; then echo "$FACTORIAL_ROOT/real_sets/tseed${TEACHER_SEED}_rseed${r}"; else echo "$RANDOM_ROOT/tseed${TEACHER_SEED}/synthetic/cic_t_c1_ipc10_rseed${r}"; fi; }
teacher_for(){ [[ "$1" == c1 ]] && echo "$TEACHER_C1" || echo "$TEACHER_R100"; }
mapping_for(){ [[ "$1" == c1 ]] && echo "$MAP_C1" || echo "$MAP_R100"; }
heads_for(){ [[ "$1" == c1 ]] && echo 10 || echo 1000; }
temp_for(){ [[ "$1" == c1 ]] && echo "$C1_MATCH_T" || echo "$R100_MATCH_T"; }
tag_for(){ local value; value="$(temp_for "$1")"; echo "${1}_T${value/./p}"; }
for ipc in "${IPCS[@]}"; do for row in real c1; do for r in "${RSEEDS[@]}"; do [[ -d "$(source_for "$ipc" "$row" "$r")" ]] || fail "missing source ipc=$ipc row=$row r=$r"; done; done; done

relabel_one(){ local ipc="$1" row="$2" col="$3" r="$4" gpu="$5"; local src teacher map heads temp tag base final expected count=0; src="$(source_for "$ipc" "$row" "$r")"; teacher="$(teacher_for "$col")"; map="$(mapping_for "$col")"; heads="$(heads_for "$col")"; temp="$(temp_for "$col")"; tag="$(tag_for "$col")"; base="$FKD_ROOT/ipc${ipc}_${row}__${tag}_rseed${r}"; final="${base}_bs10_ipc${ipc}"; expected=$((300*ipc)); [[ -d "$final" ]] && count="$(find "$final" -type f -name 'batch_*.tar'|wc -l)"; (( count==expected )) && return; (( count>0 )) && mv "$final" "${final}.partial_$(date +%Y%m%d_%H%M%S)"; CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u "$ROOT/relabel/relabel.py" --syn-data-path "$src" --fkd-path "$base" --model-pool-dir "$(dirname "$teacher")" --teacher-model-name ResNet18 --teacher-num-classes "$heads" --teacher-mapping "$map" --marginalize-temperature "$temp" --gpu 0 --batch-size 10 --workers "$WORKERS" --persistent-workers --prefetch-factor 4 --dataset-name imagenet-nette --epochs 300 --fkd-seed 42 --seed 42 --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix > "$LOG_ROOT/relabel_ipc${ipc}_${row}__${tag}_r${r}.log" 2>&1; count="$(find "$final" -type f -name 'batch_*.tar'|wc -l)"; [[ "$count" == "$expected" ]]; }

if [[ "$STAGE" == relabel || "$STAGE" == all ]]; then
 echo "[Relabel] C1 T=$C1_MATCH_T; Random100 T=$R100_MATCH_T"
 pids=(); task=0; for ipc in "${IPCS[@]}"; do for r in "${RSEEDS[@]}"; do for row in real c1; do for col in c1 random100; do gpu="$GPU0"; ((task%2)) && gpu="$GPU1"; task=$((task+1)); relabel_one "$ipc" "$row" "$col" "$r" "$gpu" & pids+=("$!"); if (( ${#pids[@]}==2 )); then wait_jobs "${pids[@]}" || fail relabel; pids=(); fi; done; done; done; done; if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail relabel; fi
fi

post_one(){ local ipc="$1" row="$2" col="$3" r="$4" s="$5" gpu="$6"; local temp tag src fkd result eta; temp="$(temp_for "$col")"; tag="$(tag_for "$col")"; src="$(source_for "$ipc" "$row" "$r")"; fkd="$FKD_ROOT/ipc${ipc}_${row}__${tag}_rseed${r}_bs10_ipc${ipc}"; result="$RESULT_ROOT/ipc${ipc}_${row}__${tag}_rseed${r}_sseed${s}.json"; eta=1; [[ "$ipc" == 1 ]] && eta=2; [[ -f "$result" ]] && return; CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u "$ROOT/validate/train_fkd.py" --fkd-path "$fkd" --mix-type cutmix --temperature "$temp" --model ResNet18 --ipc "$ipc" --exp-name "entropy_ipc${ipc}_${row}__${tag}_t${TEACHER_SEED}_r${r}_s${s}" --original-data-path "$src" --output-dir "$POST_ROOT" --batch-size 10 --epochs 300 --dataset-name imagenet-nette --gradient-accumulation-steps 2 --cos --workers "$WORKERS" --fkd_seed 42 --adamw-weight-decay 0.01 --adamw-lr-override 0.0005 --eta-override "$eta" --train-seed "$s" --persistent-workers --val-dir "$VAL_DIR" --disable-wandb --per-class-output "$result" > "$LOG_ROOT/post_ipc${ipc}_${row}__${tag}_r${r}_s${s}.log" 2>&1; }
if [[ "$STAGE" == post || "$STAGE" == all ]]; then
 echo "[Post] entropy-matched labels"
 pids=(); task=0; for ipc in "${IPCS[@]}"; do for r in "${RSEEDS[@]}"; do for row in real c1; do for col in c1 random100; do for s in "${SSEEDS[@]}"; do gpu="$GPU0"; ((task%2)) && gpu="$GPU1"; task=$((task+1)); post_one "$ipc" "$row" "$col" "$r" "$s" "$gpu" & pids+=("$!"); if (( ${#pids[@]}==4 )); then wait_jobs "${pids[@]}" || fail post; pids=(); fi; done; done; done; done; done; if (( ${#pids[@]} )); then wait_jobs "${pids[@]}" || fail post; fi
fi
echo "Complete stage=$STAGE teacher=$TEACHER_SEED"
