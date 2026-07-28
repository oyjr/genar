#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

: "${GENAR_DATA_ROOT:?Set GENAR_DATA_ROOT to the directory containing the datasets.}"

dataset="${GENAR_DATASET:-PRAD}"
encoder="${GENAR_ENCODER:-uni}"
gpu_count="${GENAR_GPUS:-4}"
epochs="${GENAR_EPOCHS:-200}"
num_workers="${GENAR_NUM_WORKERS:-4}"
require_h100="${GENAR_REQUIRE_H100:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

preflight_args=(
  scripts/preflight.py
  --dataset "${dataset}"
  --data-root "${GENAR_DATA_ROOT}"
  --encoder "${encoder}"
  --gpus "${gpu_count}"
  --global-batch-size 64
  --max-gene-count 2000
)
if [[ "${require_h100}" == "1" ]]; then
  preflight_args+=(--require-h100)
fi
python "${preflight_args[@]}"

exec python src/main.py \
  --dataset "${dataset}" \
  --data-root "${GENAR_DATA_ROOT}" \
  --encoder "${encoder}" \
  --gpus "${gpu_count}" \
  --strategy ddp \
  --epochs "${epochs}" \
  --global-batch-size 64 \
  --num-workers "${num_workers}" \
  --lr 1e-4 \
  --max-gene-count 2000 \
  --scale-config paper \
  --prediction-mode discrete \
  --final-loss-mode gaussian_kl \
  --ablation-protocol normalized \
  --model-variant original \
  --grouping-mode kmeans \
  --seed 2021
