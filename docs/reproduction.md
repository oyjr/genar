# Reproducing GenAR experiments

This page records the commands and settings used by the public implementation.
See [data.md](data.md) first.

## Paper configuration

| Setting | Value |
|---|---:|
| genes | 200 |
| hierarchy | `(1, 4, 8, 40, 100, 200)` |
| encoder | UNI, 1,024 dimensions |
| Transformer width / layers / heads | 512 / 8 / 8 |
| count cap | 2,000 |
| vocabulary | integer tokens `0..2000` |
| optimizer | Adam |
| learning rate | `1e-4` |
| global batch size | 64 |
| epochs | 200 |
| seed | 2021 |
| final-scale objective | adaptive Gaussian soft-token KL |
| intermediate objectives | interpolated soft-label KL |
| evaluation decoder | top-1 |

The defaults live in [`src/configs.py`](../src/configs.py) and
[`src/main.py`](../src/main.py). Checkpoints store the resolved configuration,
data split, selected-gene hash, and schema version.

## Train

One GPU:

```bash
python src/main.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --encoder uni \
  --gpus 1
```

Four 80-GB H100s:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/run_paper_4xH100.sh
```

The launcher validates the data, packages, GPU count/model, count cap, global
batch size, and hierarchy before training. With four processes it uses 16
samples per process, preserving the paper's global batch size of 64.

Runs are stored under `logs/<dataset>/.../<timestamp>/`. The best Lightning
checkpoint is selected by `train_loss_final`.

## Evaluate a checkpoint

```bash
python src/inference.py \
  --ckpt-path checkpoints/best.ckpt \
  --dataset PRAD \
  --slide-id MEND145 \
  --data-root "$GENAR_DATA_ROOT" \
  --output-dir inference_results/PRAD_MEND145 \
  --save-predictions
```

Inference checks the dataset, encoder, test split, vocabulary, and gene-order
hash against the checkpoint. It writes a JSON summary, per-gene CSV, and
optional prediction arrays. Reported diagnostics include log-space
PCC-10/50/200, MSE, MAE, raw-count errors, zero recovery, sequencing-depth
correlation, expression-bin PCC, and negative-binomial NLL.

CPU inference is available with `--gpu-id -1`. A trusted old checkpoint that
contains a pickled `addict.Dict` needs `--allow-legacy-pickle`; do not use that
flag with an untrusted file.

## Controlled ablations

```bash
# Architecture-matched continuous regression
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --prediction-mode continuous --continuous-loss mse

# Remove gene-identity FiLM
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --model-variant no_film

# Hard cross-entropy at the final scale
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --final-loss-mode cross_entropy

# Remove progressive decoding
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --scale-config single

# Scale sensitivity
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --scale-config k4
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --scale-config k5
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --scale-config k7

# Random grouping control
python src/main.py --dataset PRAD --data-root "$GENAR_DATA_ROOT" \
  --grouping-mode random --grouping-seed 42
```

The scale presets are:

- `k4`: `(1, 8, 40, 200)`
- `k5`: `(1, 4, 20, 100, 200)`
- `paper`: `(1, 4, 8, 40, 100, 200)`
- `k7`: `(1, 2, 4, 8, 40, 100, 200)`

## What is needed for exact numerical reproduction

The code and paper defaults are in this repository. The following large or
licensed artifacts are not:

- the five processed datasets;
- the exact ordered 200-gene panel for each paper experiment;
- the corresponding UNI feature tensors and licensed UNI weights; and
- the trained paper checkpoints.

Someone can run the full pipeline with their own valid panel and features, but
the published table values cannot be checked bit-for-bit until those exact
artifacts are released. Replacing a missing paper artifact with a newly
generated one should be treated as a new experiment, not as reproduction of
the reported number.
