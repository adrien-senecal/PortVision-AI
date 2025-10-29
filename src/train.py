import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from .settings import MODEL_DIR, DATASET_RAW_DIR, DATASET_TILED_DIR
from .tiling_utils import preprocess_dataset_with_tiles_seg
from .class_names import get_class_names_list


# ----------------------------- Logging --------------------------------- #
def setup_logging(verbosity: int) -> None:
    """Configure root logger."""
    level = (
        logging.WARNING
        if verbosity <= 0
        else (logging.INFO if verbosity == 1 else logging.DEBUG)
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------- CLI parsing ------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11 model on the DOTA-v2 dataset with optional 1024×1024 tiling."
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--image-size", type=int, default=1024, help="Input image size (pixels)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="YOLOv11 model checkpoint to start from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device string understood by Ultralytics: 'cpu', '0', '0,1', etc.",
    )

    # Tiling
    # Python 3.9+ supports BooleanOptionalAction for --tiling / --no-tiling
    parser.add_argument(
        "--tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable preprocessing into fixed-size tiles.",
    )
    parser.add_argument(
        "--tile-size", type=int, default=1024, help="Tile side length (pixels)."
    )
    parser.add_argument(
        "--tile-overlap", type=int, default=256, help="Tile overlap (pixels)."
    )
    parser.add_argument(
        "--use-premade-tiles",
        action="store_true",
        help="Use an existing tiled dataset instead of regenerating.",
    )
    parser.add_argument(
        "--tiled-dataset-suffix",
        type=str,
        default="tiles1024",
        help="Suffix for the output tiled dataset directory.",
    )

    # Misc
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v for INFO, -vv for DEBUG; use none for WARNING).",
    )
    parser.add_argument(
        "--clear-cuda-cache",
        action="store_true",
        help="Clear CUDA cache before starting (torch.cuda.empty_cache()).",
    )

    return parser.parse_args()


# ----------------------------- Helpers -------------------------------- #
def build_tiled_yaml(
    base_yaml_dir: Path,
    tiled_root: Path,
    suffix: str,
    class_names: Optional[list[str]] = None,
) -> Path:
    """
    Create a derived YAML pointing to a tiled dataset root with YOLO-style structure.
    Returns the path to the generated YAML.
    """
    class_names = class_names or get_class_names_list()

    derived_yaml = base_yaml_dir / f"dotav2_{suffix}.yaml"
    yaml_text = [
        "# Auto-generated tiled dataset",
        f"path: {tiled_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "# Classes",
        "names:",
    ]
    yaml_text += [f"  {i}: {name}" for i, name in enumerate(class_names)]

    derived_yaml.write_text("\n".join(yaml_text) + "\n", encoding="utf-8")
    logging.info("Wrote derived dataset YAML: %s", derived_yaml)
    return derived_yaml


# ----------------------------- Training ------------------------------- #
def train_dota_v2(
    epochs: int,
    batch_size: int,
    image_size: int,
    model_name: str,
    device: str,
    tiling: bool = True,
    tile_size: int = 1024,
    tile_overlap: int = 256,
    use_premade_tiles: bool = False,
    tiled_dataset_suffix: str = "tiles1024",
) -> None:
    """Train YOLOv11 on DOTA-v2, with optional tiling preprocessing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    logging.info("Loading model: %s", model_name)
    model = YOLO(model_name)

    # Dataset configuration
    data_yaml = "dotav2.yaml"
    data_yaml_path = (Path(MODEL_DIR) / data_yaml).resolve()

    # Optional: tiling preprocessing
    if tiling:
        tiled_root = DATASET_TILED_DIR / tiled_dataset_suffix
        logging.info(
            "[tiling] src=%s | out=%s | size=%d | overlap=%d",
            DATASET_RAW_DIR,
            tiled_root,
            tile_size,
            tile_overlap,
        )

        if not use_premade_tiles:
            preprocess_dataset_with_tiles_seg(
                data_root=DATASET_RAW_DIR,
                split_names=["train", "val"],
                tile_size=tile_size,
                overlap=tile_overlap,
                out_root=tiled_root,
                min_poly_area=4.0,
                min_poly_points=3,
            )
        else:
            logging.info("[tiling] Using premade tiled dataset at: %s", tiled_root)

        # Create derived YAML targeting the tiled dataset
        data_yaml_path = build_tiled_yaml(
            base_yaml_dir=Path(MODEL_DIR),
            tiled_root=tiled_root,
            suffix=tiled_dataset_suffix,
        )

        if image_size != tile_size:
            logging.warning(
                "[tiling] Consider setting --image-size %d to match tile size (current: %d).",
                tile_size,
                image_size,
            )

    # Run name encodes key hyperparams
    run_name = (
        f"dota_v2_{Path(model_name).stem}_e{epochs}_b{batch_size}_img{image_size}"
    )
    if tiling:
        run_name += f"_tile{tile_size}_ov{tile_overlap}"

    # Train
    logging.info("Starting training: run=%s | device=%s", run_name, device)
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=str(MODEL_DIR),
        name=run_name,
        save=True,
        # Example augment knobs for aerial imagery (tune as needed):
        # mosaic=0.5, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, perspective=0.0
    )

    # Save final checkpoint with descriptive name
    final_ckpt = Path(MODEL_DIR) / f"{run_name}_final.pt"
    logging.info("Saving final model to: %s", final_ckpt)
    model.save(final_ckpt)


# -------------------------------- Main -------------------------------- #
def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if args.clear_cuda_cache and torch.cuda.is_available():
        logging.info("Clearing CUDA cache.")
        torch.cuda.empty_cache()

    try:
        train_dota_v2(
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            model_name=args.model,
            device=args.device,
            tiling=args.tiling,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            use_premade_tiles=args.use_premade_tiles,
            tiled_dataset_suffix=args.tiled_dataset_suffix,
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Error during training: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
