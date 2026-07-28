#!/usr/bin/env python3
"""Extract UNI or ResNet-18 spot embeddings from HEST-style slides."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from configs import DATASETS, ENCODER_FEATURE_DIMS

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

WSI_SUBDIR = 'wsis'
ST_SUBDIR = 'st'
SLIDE_LIST_PATH = 'processed_data/all_slide_lst.txt'
SUPPORTED_ENCODERS = ('uni', 'resnet18')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract histology embeddings for GenAR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        default='all',
        choices=['all', *DATASETS],
    )
    parser.add_argument(
        '--encoder',
        default='uni',
        choices=SUPPORTED_ENCODERS,
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path(os.environ.get('GENAR_DATA_ROOT', './data')),
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument(
        '--patch-size',
        type=int,
        default=224,
        help='Full-resolution square crop in pixels; 0 uses spot metadata',
    )
    parser.add_argument(
        '--patch-scale',
        type=float,
        default=1.0,
        help='Multiplier used only when --patch-size=0',
    )
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_slide_ids(dataset_dir: Path) -> list[str]:
    path = dataset_dir / SLIDE_LIST_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Slide list not found: {path}")
    slides = [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not slides:
        raise ValueError(f"Slide list is empty: {path}")
    if len(slides) != len(set(slides)):
        raise ValueError(f"Slide list contains duplicates: {path}")
    return slides


def load_st_data(dataset_dir: Path, slide_id: str):
    path = dataset_dir / ST_SUBDIR / f'{slide_id}.h5ad'
    if not path.is_file():
        raise FileNotFoundError(f"ST file missing for {slide_id}: {path}")
    return sc.read_h5ad(path)


def load_wsi(dataset_dir: Path, slide_id: str) -> Image.Image:
    path = dataset_dir / WSI_SUBDIR / f'{slide_id}.tif'
    if not path.is_file():
        raise FileNotFoundError(f"WSI file missing for {slide_id}: {path}")
    return Image.open(path).convert('RGB')


def resolve_spot_geometry(adata, slide_id: str) -> tuple[np.ndarray, float]:
    """Return full-resolution WSI centers and spot diameter."""
    if 'spatial' not in adata.obsm:
        raise ValueError(
            f"AnnData for {slide_id} is missing obsm['spatial'] coordinates"
        )
    coords = np.asarray(adata.obsm['spatial'])
    if (
        coords.ndim != 2
        or coords.shape[1] != 2
        or coords.shape[0] == 0
        or not np.isfinite(coords).all()
    ):
        raise ValueError(
            f"AnnData for {slide_id} has invalid spatial shape {coords.shape}"
        )

    spatial_meta = adata.uns.get('spatial', {})
    scalefactors = {}
    if isinstance(spatial_meta, dict):
        library_meta = spatial_meta.get(slide_id)
        if library_meta is None:
            library_meta = spatial_meta.get('ST')
        if library_meta is None and len(spatial_meta) == 1:
            library_meta = next(iter(spatial_meta.values()))
        if isinstance(library_meta, dict):
            candidate = library_meta.get('scalefactors', {})
            if isinstance(candidate, dict):
                scalefactors = candidate

    diameter = scalefactors.get('spot_diameter_fullres')
    if diameter is None:
        diameter = estimate_diameter_from_coords(coords)
    if not np.isfinite(diameter) or float(diameter) <= 0:
        raise ValueError(f"Invalid spot diameter for {slide_id}: {diameter}")

    # HEST spatial coordinates are already in the full-resolution WSI frame.
    return coords, float(diameter)


def estimate_diameter_from_coords(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 224.0
    sample = coords if coords.shape[0] <= 1024 else coords[:1024]
    nearest_distances = []
    for point in sample:
        distances = np.sqrt(np.square(sample - point).sum(axis=1))
        nearest_distances.append(float(np.partition(distances, 1)[1]))
    if not nearest_distances:
        return 224.0
    return max(96.0, float(np.median(nearest_distances)))


def crop_patch(
    image: Image.Image,
    center_xy: tuple[float, float],
    crop_size: int,
) -> Image.Image:
    """Crop a square patch, padding WSI boundaries with black pixels."""
    width, height = image.size
    half = crop_size / 2.0
    left = round(center_xy[0] - half)
    top = round(center_xy[1] - half)
    right = left + crop_size
    bottom = top + crop_size

    crop_left = max(left, 0)
    crop_top = max(top, 0)
    crop_right = min(right, width)
    crop_bottom = min(bottom, height)
    patch = Image.new('RGB', (crop_size, crop_size))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return patch
    region = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    patch.paste(region, (crop_left - left, crop_top - top))
    return patch


def prepare_encoder(
    encoder: str,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    """Load the requested pretrained encoder and its official transform."""
    if encoder == 'resnet18':
        from torchvision import models

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
    elif encoder == 'uni':
        import timm
        from timm.data import create_transform, resolve_data_config

        try:
            model = timm.create_model(
                'hf-hub:MahmoodLab/UNI',
                pretrained=True,
                init_values=1.0e-5,
                dynamic_img_size=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "UNI could not be loaded. Accept the UNI model terms on "
                "Hugging Face and authenticate with `hf auth login`."
            ) from exc
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    model.eval()
    model.to(device)
    return model, transform


def encode_slide(
    model: torch.nn.Module,
    transform,
    image: Image.Image,
    coords: np.ndarray,
    crop_size: int,
    batch_size: int,
    device: torch.device,
    expected_dim: int,
) -> torch.Tensor:
    """Encode all spots while preserving AnnData row order."""
    outputs: list[torch.Tensor] = []
    batch: list[torch.Tensor] = []
    with torch.inference_mode():
        for index, center in enumerate(coords):
            batch.append(
                transform(crop_patch(image, (center[0], center[1]), crop_size))
            )
            if len(batch) == batch_size or index == len(coords) - 1:
                features = model(
                    torch.stack(batch).to(device, non_blocking=True)
                )
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if (
                    not torch.is_tensor(features)
                    or features.ndim != 2
                    or features.shape[1] != expected_dim
                    or not torch.isfinite(features).all()
                ):
                    raise ValueError(
                        "Encoder returned invalid features: "
                        f"shape={getattr(features, 'shape', None)}"
                    )
                outputs.append(features.detach().float().cpu())
                batch.clear()
    return torch.cat(outputs, dim=0)


def process_slide(
    model: torch.nn.Module,
    transform,
    dataset_dir: Path,
    slide_id: str,
    encoder: str,
    batch_size: int,
    patch_size: int,
    patch_scale: float,
    device: torch.device,
) -> torch.Tensor:
    adata = load_st_data(dataset_dir, slide_id)
    coords, diameter = resolve_spot_geometry(adata, slide_id)
    crop_size = patch_size or round(diameter * patch_scale)
    crop_size = max(64, crop_size)
    image = load_wsi(dataset_dir, slide_id)
    try:
        return encode_slide(
            model,
            transform,
            image,
            coords,
            crop_size,
            batch_size,
            device,
            ENCODER_FEATURE_DIMS[encoder],
        )
    finally:
        image.close()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.patch_size < 0:
        raise ValueError("--patch-size cannot be negative")
    if args.patch_scale <= 0:
        raise ValueError("--patch-scale must be positive")

    datasets: Iterable[tuple[str, dict]]
    if args.dataset == 'all':
        datasets = DATASETS.items()
    else:
        datasets = [(args.dataset, DATASETS[args.dataset])]

    device: torch.device | None = None
    model: torch.nn.Module | None = None
    transform = None
    if not args.dry_run:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model, transform = prepare_encoder(args.encoder, device)

    for dataset_name, config in datasets:
        dataset_dir = args.data_root.resolve() / config['dir_name']
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_dir}")
        output_dir = (
            dataset_dir / 'processed_data' / f'spot_features_{args.encoder}'
        )
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        for slide_id in tqdm(
            load_slide_ids(dataset_dir),
            desc=dataset_name,
            unit='slide',
        ):
            output_path = output_dir / f'{slide_id}_{args.encoder}.pt'
            if args.skip_existing and output_path.exists():
                continue
            if args.dry_run:
                print(f"[DRY-RUN] {dataset_name}: {slide_id}")
                continue
            if model is None or transform is None or device is None:
                raise RuntimeError("Encoder was not initialized")
            features = process_slide(
                model,
                transform,
                dataset_dir,
                slide_id,
                args.encoder,
                args.batch_size,
                args.patch_size,
                args.patch_scale,
                device,
            )
            torch.save(features, output_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
