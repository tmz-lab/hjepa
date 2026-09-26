#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/.stable-wm}"

task="$1"
declare -A epoch=([tworoom]=8 [reacher]=6 [pusht]=10 [cube]=2)
ckpt="${2:-$STABLEWM_HOME/hjepa/$task/hjepa_epoch_${epoch[$task]}_object.ckpt}"
out="eval_$(date +%Y%m%d_%H%M%S).txt"

for offset in $(seq 0 50 450)
do
  python eval.py -cn "$task" policy="$ckpt" eval.batch_offset="$offset" output.filename="$out"
done

grep -o "success_rate': [0-9.]*" "$(dirname "$ckpt")/$out" | awk -v task="$task" '{s += $2} END {printf "%s success over %d episodes: %.2f%%\n", task, 50 * NR, s / NR}'
