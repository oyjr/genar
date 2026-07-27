"""Extract ResNet18 embeddings for HEST-style spatial transcriptomics datasets.

The script mirrors the one-spot feature layout used by the existing UNI/Conch
embeddings so that a new `resnet18` encoder can be plugged into the training
pipeline without further code changes.  For each slide it:

1. Loads spot coordinates from the corresponding `.h5ad` file.
2. Crops RGB patches from the whole-slide image around every spot.
3. Runs the patches through an ImageNet-pretrained ResNet18 backbone.
4. Stores the resulting `[num_spots, 512]` tensor under
   `processed_data/spot_features_resnet18/{slide_id}_resnet18.pt`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from PIL import Image, ImageFile
from torchvision import models, transforms
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from configs import DATASETS  # noqa: E402

# Allow extremely large TIFF files and avoid aborting on slightly truncated data
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


OUTPUT_SUBDIR = "processed_data/spot_features_resnet18"
WSI_SUBDIR = "wsis"
ST_SUBDIR = "st"
SLIDE_LIST_PATH = "processed_data/all_slide_lst.txt"


# ---------------------------------------------------------------------------
# Argument parsing and basic I/O helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ResNet18 embeddings for HEST-style datasets",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASETS.keys()],
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GENAR_DATA_ROOT", "./data")),
        help="Root directory containing dataset folders "
             "(default: $GENAR_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device string, e.g. cuda, cuda:0, cpu (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of patches processed per forward pass (default: 128)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help=(
            "Full-resolution square crop size in pixels "
            "(paper default: 224; 0 = infer from scalefactors)"
        ),
    )
    parser.add_argument(
        "--patch-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the inferred spot diameter (default: 1.0)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip slides that already have an extracted embedding file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the slides that would be processed without writing output",
    )
    return parser.parse_args()


def load_slide_ids(dataset_dir: Path) -> List[str]:
    slide_path = dataset_dir / SLIDE_LIST_PATH
    if not slide_path.exists():
        raise FileNotFoundError(f"Slide list not found: {slide_path}")
    with slide_path.open("r", encoding="utf-8") as handle:
        slides = [line.strip() for line in handle if line.strip()]
    if not slides:
        raise RuntimeError(f"No slide IDs discovered in {slide_path}")
    return slides


def load_st_data(dataset_dir: Path, slide_id: str):
    st_path = dataset_dir / ST_SUBDIR / f"{slide_id}.h5ad"
    if not st_path.exists():
        raise FileNotFoundError(f"ST file missing for {slide_id}: {st_path}")
    return sc.read_h5ad(st_path)


def load_wsi(dataset_dir: Path, slide_id: str) -> Image.Image:
    wsi_path = dataset_dir / WSI_SUBDIR / f"{slide_id}.tif"
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI file missing for {slide_id}: {wsi_path}")
    return Image.open(wsi_path).convert("RGB")


def ensure_output_dir(dataset_dir: Path) -> Path:
    out_dir = dataset_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def resolve_spot_geometry(adata, slide_id: str) -> Tuple[np.ndarray, float]:
    """Return full-resolution WSI centers and the full-resolution spot diameter."""
    if "spatial" not in adata.obsm:
        raise ValueError(f"AnnData for {slide_id} is missing obsm['spatial'] coordinates")
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(
            f"AnnData for {slide_id} has invalid spatial shape {coords.shape}"
        )

    spatial_meta = adata.uns.get("spatial", {})
    scalefactors = {}
    if isinstance(spatial_meta, dict):
        library_meta = spatial_meta.get(slide_id)
        if library_meta is None:
            library_meta = spatial_meta.get("ST")
        if library_meta is None and len(spatial_meta) == 1:
            library_meta = next(iter(spatial_meta.values()))
        if isinstance(library_meta, dict):
            candidate = library_meta.get("scalefactors", {})
            if isinstance(candidate, dict):
                scalefactors = candidate

    fullres_diameter = scalefactors.get("spot_diameter_fullres")

    if fullres_diameter is None:
        fullres_diameter = estimate_diameter_from_coords(coords)

    # HEST stores obsm["spatial"] in the coordinate system of the
    # full-resolution WSI. ``tissue_hires_scalef`` is only for rendering the
    # derived hires image and must not be applied to these centers.
    return coords, float(fullres_diameter)


def estimate_diameter_from_coords(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 224.0
    sample = coords if coords.shape[0] <= 1024 else coords[:1024]
    dists = []
    for point in sample:
        diff = sample - point
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        if dist.size <= 1:
            continue
        nearest = np.partition(dist, 1)[1]
        dists.append(nearest)
    if not dists:
        return 224.0
    median = float(np.median(dists))
    return max(96.0, median)


# ---------------------------------------------------------------------------
# Image preprocessing and feature extraction
# ---------------------------------------------------------------------------

def build_transform(target_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def prepare_model(device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def crop_patch(image: Image.Image, center_xy: Tuple[float, float], crop_size: int) -> Image.Image:
    width, height = image.size
    half = crop_size / 2.0
    center_x, center_y = center_xy
    left = round(center_x - half)
    top = round(center_y - half)
    right = left + crop_size
    bottom = top + crop_size

    crop_left = max(left, 0)
    crop_top = max(top, 0)
    crop_right = min(right, width)
    crop_bottom = min(bottom, height)

    patch = Image.new("RGB", (crop_size, crop_size))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return patch

    region = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    paste_x = crop_left - left
    paste_y = crop_top - top
    patch.paste(region, (paste_x, paste_y))
    return patch


def encode_slide(
    model: torch.nn.Module,
    transform: transforms.Compose,
    image: Image.Image,
    coords_fullres: np.ndarray,
    crop_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    cache: List[torch.Tensor] = []
    total = coords_fullres.shape[0]

    with torch.no_grad():
        for idx in range(total):
            # Coordinates are stored as (x, y) pixel positions
            patch = crop_patch(image, (coords_fullres[idx, 0], coords_fullres[idx, 1]), crop_size)
            cache.append(transform(patch))
            if len(cache) == batch_size or idx == total - 1:
                batch = torch.stack(cache).to(device, non_blocking=True)
                emb = model(batch)
                features.append(emb.cpu())
                cache.clear()

    return torch.cat(features, dim=0)


def process_slide(
    model: torch.nn.Module,
    dataset_dir: Path,
    slide_id: str,
    batch_size: int,
    patch_size_override: int,
    patch_scale: float,
    device: torch.device,
) -> torch.Tensor:
    adata = load_st_data(dataset_dir, slide_id)
    coords_fullres, inferred_diameter = resolve_spot_geometry(adata, slide_id)

    crop_size = (
        patch_size_override
        if patch_size_override > 0
        else round(inferred_diameter * patch_scale)
    )
    if crop_size <= 0:
        crop_size = 224
    crop_size = max(64, crop_size)

    transform = build_transform(224)

    image = load_wsi(dataset_dir, slide_id)
    try:
        embeddings = encode_slide(
            model=model,
            transform=transform,
            image=image,
            coords_fullres=coords_fullres,
            crop_size=crop_size,
            batch_size=batch_size,
            device=device,
        )
    finally:
        image.close()

    return embeddings.float()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.patch_size < 0:
        raise ValueError("--patch-size cannot be negative")
    if args.patch_scale <= 0:
        raise ValueError("--patch-scale must be positive")

    if args.dataset == "all":
        dataset_sequence: Iterable[Tuple[str, dict]] = DATASETS.items()
    else:
        dataset_sequence = [(args.dataset, DATASETS[args.dataset])]

    data_root = args.data_root.resolve()
    device: Optional[torch.device] = None
    model: Optional[torch.nn.Module] = None
    if not args.dry_run:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        model = prepare_model(device)

    summary: List[Tuple[str, int]] = []

    for dataset_name, config in dataset_sequence:
        dataset_dir = data_root / config["dir_name"]
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_dir}")
        slides = load_slide_ids(dataset_dir)
        out_dir = (
            dataset_dir / OUTPUT_SUBDIR
            if args.dry_run
            else ensure_output_dir(dataset_dir)
        )
        processed = 0

        with tqdm(slides, desc=dataset_name, unit="slide") as iterator:
            for slide_id in iterator:
                output_path = out_dir / f"{slide_id}_resnet18.pt"
                iterator.set_postfix_str(slide_id)

                if args.skip_existing and output_path.exists():
                    continue
                if args.dry_run:
                    print(f"[DRY-RUN] {dataset_name}: would process {slide_id}")
                    continue
                if model is None or device is None:
                    raise RuntimeError("Extractor model/device was not initialized")

                embeddings = process_slide(
                    model=model,
                    dataset_dir=dataset_dir,
                    slide_id=slide_id,
                    batch_size=args.batch_size,
                    patch_size_override=args.patch_size,
                    patch_scale=args.patch_scale,
                    device=device,
                )
                torch.save(embeddings, output_path)
                processed += 1

        summary.append((dataset_name, processed))

    if args.dry_run:
        print("Dry run finished — no files were written.")
    else:
        print("Extraction summary:")
        for dataset_name, count in summary:
            print(f"  - {dataset_name}: {count} slides processed")


if __name__ == "__main__":
    main()
