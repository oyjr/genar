# Data preparation

GenAR reads one AnnData file and one histology-feature tensor per slide. The
repository does not download patient data or redistribute UNI/CONCH weights.

## 1. Get the slides

The five paper datasets are available through their original sources and the
[HEST collection](https://github.com/mahmoodlab/HEST). Follow the dataset terms
and keep the files outside the Git repository.

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
[UNI](https://github.com/mahmoodlab/UNI) features. CONCH features are
512-dimensional and must follow the
[CONCH](https://github.com/mahmoodlab/CONCH) license and preprocessing recipe.
These weights are not included here.

For the ResNet-18 encoder comparison, the repository includes a complete
ImageNet extractor:

```bash
python src/preprocess/extract_resnet18_embeddings.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --device cuda \
  --batch-size 128
```

Each output must be a finite `float32` tensor shaped
`[number_of_spots, encoder_dimension]`. Keep the AnnData row order unchanged.

## 4. Build the gene hierarchy

Start with the intended 200-gene panel in
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

The initial gene panel is an experimental input, not something the clustering
script guesses. To reproduce a published table exactly, use the same panel,
slide files, and feature tensors as that experiment.

## 5. Validate before training

This check works on a CPU machine and does not start a job:

```bash
python scripts/preflight.py \
  --dataset PRAD \
  --data-root "$GENAR_DATA_ROOT" \
  --encoder uni \
  --data-only
```

It fails if a required slide, feature file, gene list, split, hierarchy field,
or provenance hash is missing or inconsistent.
