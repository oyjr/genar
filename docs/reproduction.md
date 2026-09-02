# Training

Prepare the inputs as described in [Data preparation](data.md), then start a training run:

```bash
python src/main.py \
  --dataset PRAD \
  --data-root data \
  --encoder uni \
  --gpus 1
```

Use `--gpus 4` for four-GPU training. Runs and checkpoints are written under `logs/`.

## Ablations

Continuous regression:

```bash
python src/main.py \
  --dataset PRAD \
  --data-root data \
  --prediction-mode continuous \
  --continuous-loss mse
```

Remove gene-identity FiLM:

```bash
python src/main.py \
  --dataset PRAD \
  --data-root data \
  --model-variant no_film
```

Use hard cross-entropy at the final scale:

```bash
python src/main.py \
  --dataset PRAD \
  --data-root data \
  --final-loss-mode cross_entropy
```

Change the decoding hierarchy or use random gene grouping:

```bash
python src/main.py --dataset PRAD --data-root data --scale-config k4
python src/main.py --dataset PRAD --data-root data --scale-config k5
python src/main.py --dataset PRAD --data-root data --scale-config k7
python src/main.py --dataset PRAD --data-root data \
  --grouping-mode random --grouping-seed 42
```

Use `python src/main.py --help` for the complete set of training options.
