#!/usr/bin/env bash
set -e
STEPS=${1:-100}
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/venv/bin/python"; [ -x "$PY" ] || PY=python3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT/repo"
"$PY" trainer.py --dataset_path "$ROOT/train/ds" --checkpoint_dir "$ROOT/models/ace" \
  --exp_name sifeng_lora --precision bf16-mixed --max_steps "$STEPS" \
  --every_n_train_steps 25 --accumulate_grad_batches 2 \
  --lora_config_path "$ROOT/train/sifeng_lora.json" --logger_dir "$ROOT/train/logs" \
  --num_workers 8
