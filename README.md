# GenAR

GenAR is a spatial transcriptomics modeling codebase for predicting gene
expression from histology image features using next-scale autoregressive models.

## Installation

Recommended Python version: 3.10.

```bash
pip install -r requirements.txt
```



## Datasets

We use HEST datasets. Please refer to HEST-1k instructions to download the data:

- https://huggingface.co/datasets/MahmoodLab/hest


## Directory Structure


```
genar/
  data/
    PRAD/
      wsis/
      st/
    her2st/
      wsis/
      st/
```

After preprocessing:

```
genar/
  data/
    PRAD/
      wsis/
      st/
      processed_data/
        spot_features_uni/
        selected_gene_list.txt
        all_slide_lst.txt
    her2st/
      wsis/
      st/
      processed_data/
```

Configure the root with `--data-root` or the `GENAR_DATA_ROOT` environment
variable. For preprocessing that loads h5ad files, set `GENAR_H5AD_ROOT` (or
pass `--h5ad-root`) to the directory containing `<slide_id>.h5ad` files.

## Preprocessing

Gene clustering (reorders selected genes):

```bash
python src/preprocess/run_clustering.py --dataset PRAD --data-root ./data \
  --h5ad-root /path/to/h5ad
```

Extract ResNet18 embeddings (optional, for `resnet18` encoder):

```bash
python src/preprocess/extract_resnet18_embeddings.py --dataset PRAD \
  --data-root ./data --device cuda --batch-size 128
```

## Training

```bash
python src/main.py --dataset PRAD --data-root ./data --gpus 1
```

Key dataset arguments (see `src/main.py` for full options):

- `--dataset`: PRAD, her2st, kidney, mouse_brain, ccRCC
- `--data-root`: root directory containing datasets
- `--encoder`: uni, conch, resnet18

## Inference

```bash
python src/inference.py \
  --ckpt_path logs/PRAD/GENAR/best-epoch=epoch=01-loss=...ckpt \
  --dataset PRAD \
  --slide_id MEND144 \
  --data-root ./data \
  --output_dir ./inference_results
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```
@article{ouyang2025genar,
  title={GenAR: Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction},
  author={Ouyang, Jiarui and Wang, Yihui and Gao, Yihang and Xu, Yingxue and Yang, Shu and Chen, Hao},
  journal={arXiv preprint arXiv:2510.04315},
  year={2025}
}
```
