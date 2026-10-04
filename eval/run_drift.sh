#!/usr/bin/env bash
# Both drift metrics for a folder of videos: Color Drift (CPU) and Motion Drift (GPU, VBench + RAFT).
#
#   bash eval/run_drift.sh outputs/self_forcing
#   GPUS=0,1,2,3 bash eval/run_drift.sh outputs/longlive
#   END=1917 bash eval/run_drift.sh outputs/sf_240s        # 4 min clips scored over their first 2 min
#
# Writes <videos_dir>_color_drift.csv and <videos_dir>_motion_drift.csv next to the folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VID="${1:?usage: run_drift.sh <videos_dir>}"
VID="${VID%/}"
PYTHON="${PYTHON:-python}"                            # an environment with vbench (see README)
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"            # comma-separated GPU ids for motion drift
END="${END:-0}"                                       # 0: whole clip
OUT="${OUT:-$VID}"                                    # output prefix
IFS=',' read -r -a GPU <<< "$GPUS"

echo "== Color Drift: $VID"
"$PYTHON" "$HERE/color_drift.py" "$VID" --end "$END" --out "${OUT}_color_drift.csv"

echo "== Motion Drift: $VID  (GPU ${GPUS})"
pids=(); parts=()
for i in "${!GPU[@]}"; do
  part="${OUT}_motion_drift.csv.part$i"
  CUDA_VISIBLE_DEVICES="${GPU[$i]}" "$PYTHON" "$HERE/motion_drift.py" "$VID" --end "$END" \
    --shard "$i" --num_shards "${#GPU[@]}" --out "$part" > "${OUT}_motion_drift.log$i" 2>&1 &
  pids+=($!); parts+=("$part")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] || { echo "a motion-drift shard failed; see ${OUT}_motion_drift.log*" >&2; exit 1; }
"$PYTHON" "$HERE/motion_drift.py" --merge "${OUT}_motion_drift.csv" "${parts[@]}"
rm -f "${parts[@]}" "${OUT}"_motion_drift.log*
