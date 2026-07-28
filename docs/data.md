# Data preparation

GenAR expects one AnnData file and one histology-feature tensor per slide.
Download the datasets and pretrained model weights from their original
sources.

## 1. Get the slides

The five datasets used in the paper are available from their original sources
and through the [HEST collection](https://github.com/mahmoodlab/HEST). Follow
the terms of each dataset and keep the data outside the Git repository.

Arrange each dataset like this:

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
      spot_features_uni/
        MEND145_uni.pt
        ...
```

Use `her2st`, `kidney`, `mouse_brain`, or `ccRCC` in place of `PRAD` for the
other datasets. File names are derived from the slide IDs in
`all_slide_lst.txt`.

## 2. Check the AnnData files

For every slide:

- `adata.X` must contain raw, non-negative integer counts;
- gene symbols must be in `adata.var_names`;
- coordinates must be in `adata.obsm["spatial"]` or
  `adata.obsm["positions"]`; and
- row order must match the feature tensor exactly.

The default held-out slides are defined in
[`src/configs.py`](../src/configs.py): MEND145, SPA148, NCBI697, NCBI667, and
INT2 for PRAD, HER2ST, Kidney, Mouse Brain, and ccRCC, respectively.

## 3. Prepare histology features

The paper configuration uses 1,024-dimensional
[UNI](https://huggingface.co/MahmoodLab/UNI) features. Accept the model terms,
authenticate once, and run:

```bash
hf auth login
python src/preprocess/extract_embeddings.py \
  --dataset PRAD \
  --encoder uni \
  --data-root "$GENAR_DATA_ROOT" \
  --device cuda \
  --batch-size 128
```

For the ResNet-18 encoder comparison, change `--encoder uni` to
`--encoder resnet18`. CONCH features are 512-dimensional and must be prepared
with the [official CONCH](https://github.com/mahmoodlab/CONCH) preprocessing
and license terms; CONCH weights are not included here.

Each output must be a finite `float32` tensor shaped
`[number_of_spots, encoder_dimension]`. Keep the AnnData row order unchanged.
The extractor writes tensor-only `.pt` files that can be loaded without
enabling Python pickle objects.

## 4. Build the gene hierarchy

Provide the 200-gene panel for the experiment in
`processed_data/selected_gene_list.txt`, one gene symbol per line. The command
below uses training slides only, saves the original list, and rewrites the
working list in coarse-to-fine order:

```bash
python src/preprocess/run_clustering.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --h5ad-root "$GENAR_DATA_ROOT/PRAD/st" \
  --seed 42
```

It creates:

- `unclustered_selected_gene_list.txt`, the input order;
- `selected_gene_list.txt`, the clustered order; and
- `clustering_info.json`, including the split, parameters, permutation, and
  hashes.

`run_clustering.py` reorders the supplied panel; it does not select the genes.
Use the same panel, slide files, and feature tensors when comparing with a
reported experiment.

## 5. Validate before training

This check runs on CPU:

```bash
python scripts/preflight.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --encoder uni \
  --data-only
```

The command reports missing slides, feature files, gene lists, split metadata,
hierarchy fields, and mismatched hashes.
