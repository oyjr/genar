# Data preparation

## Download HEST data

Download one registered dataset with:

```bash
python scripts/download_hest.py --dataset PRAD
```

Use `all` to download all five datasets. The command creates the data needed by GenAR:

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
```

`all_slide_lst.txt` contains the downloaded slide IDs. `selected_gene_list.txt` contains the ordered gene panel paired with the released GenAR checkpoint.

The other dataset directories are `her2st`, `kidney`, `mouse_brain`, and `ccRCC`.

## Extract UNI features

UNI weights and precomputed features are not included. Accept the [UNI model terms](https://huggingface.co/MahmoodLab/UNI), authenticate, and extract one feature tensor per slide:

```bash
hf auth login
python src/preprocess/extract_embeddings.py \
  --dataset PRAD \
  --encoder uni \
  --data-root data
```

The tensors are written under:

```text
data/PRAD/processed_data/spot_features_uni/
```

Each AnnData file must contain non-negative count data, gene symbols in `adata.var_names`, and spatial coordinates in `adata.obsm["spatial"]` or `adata.obsm["positions"]`.

## Train with a different gene panel

For a new experiment, place one gene symbol per line in `processed_data/selected_gene_list.txt`, then build its coarse-to-fine ordering:

```bash
python src/preprocess/run_clustering.py \
  --dataset PRAD \
  --data-root data \
  --h5ad-root data/PRAD/st \
  --seed 42
```
