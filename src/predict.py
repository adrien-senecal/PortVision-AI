import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from .settings import MODEL_DIR, PREDICTIONS_DIR
from .class_names import load_class_names


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict object classes using a trained YOLOv11 model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model file relative to MODEL_DIR (e.g. 'dotav2_yolo11n_e10_b8_img416_final.pt').",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image path relative to PREDICTIONS_DIR or an absolute path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for inference: 'cpu' or GPU index (e.g., '0'). Default: '0'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Subdirectory to save results under PREDICTIONS_DIR/results/. If omitted, nothing is saved.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[9, 10, 15],
        help="Class IDs to predict (space-separated). Default: 9 10 15.",
    )

    # Tiled inference
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable tiled inference with stitching.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size in pixels (default: 1024).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Tile overlap in pixels (default: 256).",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------- #
# Inference entry point
# ----------------------------------------------------------------------------- #


def predict_image(
    model_path: Union[str, Path],
    image_path: Union[str, Path],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    output_dir: Optional[str],
    classes: Optional[Sequence[int]] = None,
    tile: bool = False,
    tile_size: int = 1024,
    overlap: int = 256,
):
    """
    Predict objects in an image using a YOLO model.

    Returns:
        - If tile=False: Ultralytics Results list.
        - If tile=True: numpy.ndarray of detections [x1, y1, x2, y2, conf, cls].
    """
    # Resolve paths
    model_path = (
        (MODEL_DIR / model_path)
        if not Path(model_path).is_absolute()
        else Path(model_path)
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = PREDICTIONS_DIR / img_path
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Load model and class names
    logging.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))
    class_names = load_class_names()

    # Output handling
    save_results = output_dir is not None
    project_path: Optional[Path] = None
    if save_results:
        project_path = PREDICTIONS_DIR / output_dir
        project_path.mkdir(parents=True, exist_ok=True)
        logging.info("Saving results under: %s", project_path)

    if tile:
        return _predict_image_tiled(
            model=model,
            image_path=str(img_path),
            class_names=class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            save_results=save_results,
            project_path=project_path,
            tile_size=tile_size,
            overlap=overlap,
            classes=classes or [],
        )

    # Single-shot prediction
    logging.info("Running single-shot inference on: %s", img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Handle grayscale or alpha
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # Drop alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        save=save_results,
        project=str(project_path) if project_path else None,
        name="prediction" if save_results else None,
        classes=list(classes) if classes else None,
        verbose=False,
    )

    # Pretty print results to console
    _print_results(results, class_names)

    # Save stitched.png and detections.csv if save_results is True
    if save_results and project_path is not None:
        # Extract detections from results
        all_dets: List[List[float]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(np.int32)

            for b, score, c in zip(xyxy, confs, clss):
                if float(score) < conf_threshold:
                    continue
                all_dets.append(
                    [
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3]),
                        float(score),
                        int(c),
                    ]
                )

        if all_dets:
            dets = np.array(all_dets, dtype=np.float32)
            vis = _draw_dets(img, dets, class_names)
            pred_dir = project_path / "prediction"
            pred_dir.mkdir(parents=True, exist_ok=True)

            out_img = pred_dir / "stitched.png"
            out_txt = pred_dir / "detections.csv"

            cv2.imwrite(str(out_img), vis)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("x1,y1,x2,y2,conf,class\n")
                for x1, y1, x2, y2, conf, cls in dets:
                    f.write(
                        f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{float(conf):.4f},{int(cls)}\n"
                    )

            logging.info("Saved stitched visualization to: %s", out_img)
            logging.info("Saved detections to: %s", out_txt)

    return results


# ----------------------------------------------------------------------------- #
# Tiled inference helpers
# ----------------------------------------------------------------------------- #


def _tile_windows(
    W: int, H: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Generate sliding-window tile coordinates over an image of size (W, H)."""
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0]

    windows: List[Tuple[int, int, int, int]] = []
    for ty in ys:
        for tx in xs:
            x2 = min(tx + tile_size, W)
            y2 = min(ty + tile_size, H)
            x1 = x2 - tile_size if (x2 - tx) < tile_size else tx
            y1 = y2 - tile_size if (y2 - ty) < tile_size else ty
            windows.append((x1, y1, x1 + tile_size, y1 + tile_size))
    return windows


def _nms_per_class(dets: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Apply per-class Non-Maximum Suppression.

    Args:
        dets: (N, 6) array with columns [x1, y1, x2, y2, conf, cls]
        iou_thresh: IoU threshold

    Returns:
        Indices to keep (np.ndarray[int]).
    """
    if dets.size == 0:
        return np.array([], dtype=int)

    keep_indices: List[int] = []
    classes = np.unique(dets[:, 5].astype(np.int32))
    for c in classes:
        idxs = np.where(dets[:, 5].astype(np.int32) == c)[0]
        boxes = dets[idxs, 0:4]
        scores = dets[idxs, 4]
        order = scores.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep_indices.append(idxs[i])
            if order.size == 1:
                break
            ious = _iou(boxes[i], boxes[order[1:]])
            remaining = np.where(ious <= iou_thresh)[0]
            order = order[remaining + 1]

    return np.array(keep_indices, dtype=int)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter = w * h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-12
    return inter / union


def _draw_dets(
    image: np.ndarray, dets: np.ndarray, class_names: Dict[int, str]
) -> np.ndarray:
    """Draw detections on an image."""
    img = image.copy()
    for x1, y1, x2, y2, conf, cls in dets:
        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
        cls_i = int(cls)
        cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
        label = f"{class_names.get(cls_i, str(cls_i))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img, (x1_i, y1_i - th - 4), (x1_i + tw + 2, y1_i), (0, 255, 0), -1
        )
        cv2.putText(
            img,
            label,
            (x1_i + 1, y1_i - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def _predict_image_tiled(
    model: YOLO,
    image_path: str,
    class_names: Dict[int, str],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    save_results: bool,
    project_path: Optional[Path],
    tile_size: int,
    overlap: int,
    classes: Sequence[int],
) -> np.ndarray:
    """Run tiled inference and merge detections with per-class NMS."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = image.shape[:2]

    logging.info(
        "Running tiled inference (%dx%d, overlap=%d) on %s",
        tile_size,
        tile_size,
        overlap,
        image_path,
    )

    all_dets: List[List[float]] = []
    for x1, y1, x2, y2 in _tile_windows(W, H, tile_size, overlap):
        tile_img = image[y1:y2, x1:x2]
        results = model.predict(
            source=tile_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            save=False,
            verbose=False,
            classes=list(classes) if classes else None,
        )
        if not results:
            continue
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(np.int32)

        for b, score, c in zip(xyxy, confs, clss):
            if float(score) < conf_threshold:
                continue
            gx1, gy1, gx2, gy2 = (
                float(b[0] + x1),
                float(b[1] + y1),
                float(b[2] + x1),
                float(b[3] + y1),
            )
            all_dets.append([gx1, gy1, gx2, gy2, float(score), int(c)])

    if not all_dets:
        logging.info("No objects detected.")
        return np.empty((0, 6), dtype=np.float32)

    dets = np.array(all_dets, dtype=np.float32)
    keep = _nms_per_class(dets, iou_threshold)
    dets_nms = dets[keep]

    _print_dets_array(dets_nms, class_names)

    if save_results and project_path is not None:
        vis = _draw_dets(image, dets_nms, class_names)
        pred_dir = project_path / "prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)

        out_img = pred_dir / "stitched.png"
        out_txt = pred_dir / "detections.csv"

        cv2.imwrite(str(out_img), vis)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("x1,y1,x2,y2,conf,class\n")
            for x1, y1, x2, y2, conf, cls in dets_nms:
                f.write(
                    f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{float(conf):.4f},{int(cls)}\n"
                )

        logging.info("Saved stitched visualization to: %s", out_img)
        logging.info("Saved detections to: %s", out_txt)

    return dets_nms


# ----------------------------------------------------------------------------- #
# Output formatting
# ----------------------------------------------------------------------------- #


def _print_results(results, class_names: Dict[int, str]) -> None:
    """Pretty-print Ultralytics results to the console."""
    if not results:
        logging.info("No detections.")
        return

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            logging.info("No objects detected.")
            continue

        logging.info("Found %d objects:", len(boxes))
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names.get(class_id, f"Unknown class {class_id}")
            logging.info(
                "  #%d | Class: %s (ID %d) | Conf: %.3f | Box: (%.1f, %.1f, %.1f, %.1f)",
                i + 1,
                class_name,
                class_id,
                confidence,
                x1,
                y1,
                x2,
                y2,
            )


def _print_dets_array(dets: np.ndarray, class_names: Dict[int, str]) -> None:
    """Pretty-print [x1, y1, x2, y2, conf, cls] detections."""
    logging.info("Found %d objects after stitching:", len(dets))
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(dets):
        class_name = class_names.get(int(cls), f"Unknown class {int(cls)}")
        logging.info(
            "  #%d | Class: %s (ID %d) | Conf: %.3f | Box: (%.1f, %.1f, %.1f, %.1f)",
            i + 1,
            class_name,
            int(cls),
            float(conf),
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        )


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #


def main() -> int:
    """Entrypoint for CLI usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()

    try:
        predict_image(
            model_path=args.model,
            image_path=args.image,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            output_dir=args.output_dir,
            classes=args.classes,
            tile=args.tile,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        if args.output_dir is not None:
            out_root = PREDICTIONS_DIR / "results" / args.output_dir
            logging.info("Prediction results saved under: %s", out_root)

    except Exception as exc:
        logging.exception("Error during prediction: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
