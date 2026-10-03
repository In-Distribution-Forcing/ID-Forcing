#!/usr/bin/env bash
# ID-Forcing on LongLive (released base model + LoRA). Default: 2-minute videos.
#
#   PROMPT="A white and orange tabby cat ..." bash run_longlive.sh      # a single prompt
#   PROMPT_FILE=prompts/moviegenbench_128.txt GPUS=0,1,2,3 bash run_longlive.sh
#   DURATION=240 SEED=333 OUTPUT=outputs/longlive_240s bash run_longlive.sh
#
# With several GPUs the prompts are split across them, one process per GPU. Clips that already
# exist are skipped, so an interrupted run can simply be started again.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
PROMPT="${PROMPT:-}"                                   # one prompt; takes precedence over PROMPT_FILE
PROMPT_FILE="${PROMPT_FILE:-}"                         # one prompt per line
DURATION="${DURATION:-120}"                            # seconds; 120 s = 160 chunks of 0.75 s
SEED="${SEED:-1356145}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"             # comma-separated GPU ids
OUTPUT="${OUTPUT:-$ROOT/outputs/longlive}"
GEN_CKPT="${GEN_CKPT:-$ROOT/checkpoints/longlive/models/longlive_base.pt}"
LORA_CKPT="${LORA_CKPT:-$ROOT/checkpoints/longlive/models/lora.pt}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a GPU <<< "$GPUS"
if [ -n "$PROMPT" ]; then
  SRC=(--prompt "$PROMPT"); GPU=("${GPU[0]}")
elif [ -n "$PROMPT_FILE" ]; then
  SRC=(--prompt_file "$PROMPT_FILE")
else
  echo "set PROMPT=\"...\" or PROMPT_FILE=<file with one prompt per line>" >&2; exit 1
fi
mkdir -p "$OUTPUT/logs"

run() {  # run <shard> <gpu>
  CUDA_VISIBLE_DEVICES="$2" "$PYTHON" "$ROOT/longlive/idforcing.py" "${SRC[@]}" \
    --seconds "$DURATION" --seed "$SEED" --generator_ckpt "$GEN_CKPT" --lora_ckpt "$LORA_CKPT" \
    --output_folder "$OUTPUT" --shard "$1" --num_shards "${#GPU[@]}"
}

if [ "${#GPU[@]}" -eq 1 ]; then
  run 0 "${GPU[0]}" 2>&1 | tee "$OUTPUT/logs/shard0.log"
else
  pids=()
  for i in "${!GPU[@]}"; do
    run "$i" "${GPU[$i]}" > "$OUTPUT/logs/shard$i.log" 2>&1 &
    pids+=($!)
    echo "shard $i -> GPU ${GPU[$i]}  (log: $OUTPUT/logs/shard$i.log)"
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || { echo "a shard failed; see $OUTPUT/logs" >&2; exit 1; }
fi
echo "done: $OUTPUT"
