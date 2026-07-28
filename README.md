# GenAR

Official PyTorch implementation of **GenAR: Next-scale autoregressive
generation for spatial gene expression prediction**, published in *Medical
Image Analysis* (Volume 114, Article 104232).

[Paper](https://doi.org/10.1016/j.media.2026.104232) ·
[Data preparation](docs/data.md) ·
[Experiment settings](docs/reproduction.md) ·
[License](LICENSE)

GenAR predicts raw spatial gene-expression counts from an H&E feature vector
and the spot coordinates. It groups genes from coarse to fine, then generates
integer count tokens through the hierarchy instead of regressing each gene
independently.

## Requirements

The repository includes the model, preprocessing, training, inference,
evaluation, and ablation code. Spatial transcriptomics data, UNI/CONCH weights,
precomputed features, and trained checkpoints are not distributed here.

For a training run you need:

- Python 3.10;
- a CUDA GPU (the paper runs used NVIDIA H100 80-GB GPUs);
- raw-count AnnData files and matching spatial coordinates;
- one histology-feature tensor per slide; and
- the ordered 200-gene panel used by the experiment.

See [docs/data.md](docs/data.md) for the expected directory layout and file
formats.

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

The environment pins PyTorch 2.13.0 and PyTorch Lightning 2.6.1. If the default
PyTorch wheel does not match your CUDA driver, install the 2.13.0 build for
your CUDA runtime first, then install the remaining requirements.

Check the installation:

```bash
python -m unittest discover -s tests -v
```

## Prepare the data

Choose a directory for the datasets:

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
    wsis/
      MEND145.tif
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

The paper uses UNI features. After accepting the
[UNI model terms](https://huggingface.co/MahmoodLab/UNI), extract them with:

```bash
hf auth login
python src/preprocess/extract_embeddings.py \
  --dataset PRAD \
  --encoder uni \
  --data-root "$GENAR_DATA_ROOT"
```

Check the prepared files before starting training:

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

Four H100s:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/run_paper_4xH100.sh
```

The script checks the environment and data files, then launches four processes
with 16 samples per process, preserving the global batch size of 64. Each run
is written to a timestamped directory under `logs/`.

The published held-out slide is used only for final evaluation. By default
there is no validation loader; checkpoints are selected by
`train_loss_final`. A separate validation split can be supplied explicitly
with `--val-slides`.

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

Before inference, the saved dataset, encoder, held-out split, vocabulary, and
gene order are compared with the local inputs. The command writes a JSON
summary, per-gene statistics, and optional prediction arrays. Checkpoints and
feature tensors are loaded in tensor-only mode. Use `--gpu-id -1` for CPU
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

## Reproducing the reported results

The repository contains the implementation and the settings reported in the
paper. Reproducing the published numbers also requires the processed slides,
the gene order used for each dataset, the corresponding feature tensors, and
the trained checkpoints. These files are not included in this release. Results
obtained with a different gene panel or newly extracted features should be
reported as a separate experiment.

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

GenAR is released under the [MIT License](LICENSE). Dataset licenses and the
licenses for UNI, CONCH, and other pretrained models apply separately.
