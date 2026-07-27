# GenAR

Official code for **GenAR: Next-scale autoregressive generation for spatial
gene expression prediction**, published in *Medical Image Analysis* (Volume
114, Article 104232).

[Paper](https://doi.org/10.1016/j.media.2026.104232) ·
[Data preparation](docs/data.md) ·
[Reproduction guide](docs/reproduction.md)

GenAR predicts raw spatial gene-expression counts from an H&E feature vector
and the spot coordinates. It groups genes from coarse to fine, then generates
integer count tokens through the hierarchy instead of regressing each gene
independently.

## Before you start

This repository contains the model, preprocessing, training, inference,
metrics, ablations, and tests. It does not contain patient data, licensed
UNI/CONCH weights, precomputed paper features, or trained paper checkpoints.

For a training run you need:

- Python 3.10;
- a CUDA GPU (the paper runs used NVIDIA H100 80-GB GPUs);
- raw-count AnnData files and matching spatial coordinates;
- one histology-feature tensor per slide; and
- the ordered 200-gene panel used by the experiment.

The exact file contract is in [docs/data.md](docs/data.md).

## Install

Conda:

```bash
git clone https://github.com/oyjr/genar.git
cd genar
conda env create -f environment.yml
conda activate genar
```

Or use an existing Python 3.10 environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned release uses PyTorch 2.1.2 and PyTorch Lightning 2.1.4. If the
default PyTorch wheel does not match your CUDA driver, install the appropriate
PyTorch build first, then install the remaining requirements.

Check the installation:

```bash
python -m unittest discover -s tests -v
```

## Prepare the data

Set one portable root; no source file contains a machine-specific data path:

```bash
export GENAR_DATA_ROOT="$PWD/data"
```

Expected layout:

```text
data/
  PRAD/
    st/
      MEND145.h5ad
      ...
    processed_data/
      all_slide_lst.txt
      selected_gene_list.txt
      unclustered_selected_gene_list.txt
      clustering_info.json
      spot_features_uni/
        MEND145_uni.pt
        ...
```

Build the hierarchy from training slides:

```bash
python src/preprocess/run_clustering.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --h5ad-root "$GENAR_DATA_ROOT/PRAD/st" \
  --seed 42
```

Then check the complete data contract without needing a GPU:

```bash
python scripts/preflight.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --encoder uni \
  --data-only
```

## Train

The shortest paper-configuration command is:

```bash
python src/main.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --encoder uni \
  --gpus 1
```

For the reviewed four-H100 launch:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/run_paper_4xH100.sh
```

The launcher runs a bounded preflight and keeps the global batch size at 64
(16 samples per process). Runs go to timestamped folders under `logs/`, so an
existing checkpoint is not silently overwritten.

The paper defaults are 200 genes, scales `(1, 4, 8, 40, 100, 200)`, a count cap
of 2,000, Transformer width/layers/heads `512/8/8`, Adam at `1e-4`, 200 epochs,
and seed 2021. See [docs/reproduction.md](docs/reproduction.md) for the full
configuration and ablation commands.

## Inference

```bash
python src/inference.py \
  --ckpt-path checkpoints/best.ckpt \
  --dataset PRAD \
  --slide-id MEND145 \
  --data-root "$GENAR_DATA_ROOT" \
  --output-dir inference_results/PRAD_MEND145 \
  --save-predictions
```

The checkpoint is checked against the dataset, encoder, held-out split,
vocabulary, and selected-gene order. Inference writes a JSON summary,
per-gene statistics, and optional prediction arrays. Use `--gpu-id -1` for CPU
inference.

## Utilities

```bash
# Forward-pass FLOPs; frozen histology encoder excluded
python tools/profile_genar_flops.py --device cuda

# Count-token usage
python tools/analyze_token_usage.py predictions.npz --count-cap 2000

# Nominal gene-group enrichment
python tools/gene_group_enrichment.py \
  --groups groups.tsv \
  --reference reference_gene_sets.tsv \
  --background-size 20000 \
  --output enrichment.csv
```

## Reproduction boundary

The implementation and reported default settings are public here. Exact
numerical reproduction of the paper also needs the original ordered gene
panels, feature tensors, processed slides, and checkpoints; those artifacts are
not currently bundled. A run made with newly selected genes or newly extracted
features is a valid new run, but it should not be presented as a bit-for-bit
reproduction of a paper table.

## Citation

```bibtex
@article{ouyang2026genar,
  title   = {GenAR: Next-scale autoregressive generation for spatial gene expression prediction},
  author  = {Ouyang, Jiarui and Wang, Yihui and Gao, Yihang and Xu, Yingxue and Yang, Shu and Chen, Hao},
  journal = {Medical Image Analysis},
  volume  = {114},
  pages   = {104232},
  year    = {2026},
  doi     = {10.1016/j.media.2026.104232}
}
```

GitHub can also read the citation metadata from [CITATION.cff](CITATION.cff).

## License

No software license has been added yet. Until the authors choose one, copyright
law reserves reuse and redistribution rights. Dataset and foundation-model
licenses apply separately.
