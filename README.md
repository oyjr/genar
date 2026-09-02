# GenAR

Official PyTorch implementation of **GenAR: Next-scale autoregressive generation for spatial gene expression prediction**, published in *Medical Image Analysis*.

[Paper](https://doi.org/10.1016/j.media.2026.104232) ·
[Models](https://huggingface.co/ouyjr/GenAR) ·
[HEST](https://huggingface.co/datasets/MahmoodLab/hest) ·
[Data preparation](docs/data.md) ·
[Training](docs/reproduction.md) ·
[License](LICENSE)

GenAR predicts spatial gene-expression counts from histology features and spot coordinates using coarse-to-fine autoregressive decoding.

## Installation

```bash
git clone https://github.com/oyjr/genar.git
cd genar
python -m pip install -r requirements.txt
```

## Models and data

```bash
python scripts/download_checkpoint.py --model PRAD
python scripts/download_hest.py --dataset PRAD
```

Use `all` to download all five datasets or pretrained checkpoints.

The data command downloads the registered HEST `.h5ad` files and WSIs and prepares the required slide and gene lists. UNI weights and precomputed features are not included.

## Histology features

```bash
hf auth login
python src/preprocess/extract_embeddings.py \
  --dataset PRAD \
  --encoder uni \
  --data-root data
```

## Inference

```bash
python src/inference.py \
  --ckpt-path checkpoints/prad.ckpt \
  --dataset PRAD \
  --slide-id MEND145 \
  --data-root data \
  --output-dir inference_results/PRAD_MEND145
```

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

## License

Source code is licensed under the [MIT License](LICENSE). Dataset and encoder terms apply separately.
