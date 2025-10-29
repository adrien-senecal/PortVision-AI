from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

# =========================
# I/O des labels YOLOv8-seg
# =========================


def load_yolo_seg_labels(lbl_path: Path) -> List[Tuple[int, np.ndarray]]:
    """
    Charge un fichier de labels YOLOv8-seg.
    Retour:
      Liste de tuples (cls:int, poly: ndarray (N,2) en coords normalisées [0,1]).
    Hypothèse: 1 polygone par ligne.
    """
    instances = []
    if not lbl_path.exists():
        return instances
    with lbl_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3 or (len(parts) - 1) % 2 != 0:
                # au minimum: "c x y x y x y" => 1 classe + 3 sommets
                continue
            c = int(float(parts[0]))
            coords = np.array([float(p) for p in parts[1:]], dtype=np.float32)
            poly = coords.reshape(-1, 2)  # (N,2)
            # sécurité: enlever points NaN/inf
            if not np.isfinite(poly).all():
                continue
            instances.append((c, poly))
    return instances


def save_yolo_seg_labels(lbl_path: Path, instances: List[Tuple[int, np.ndarray]]):
    """
    Sauvegarde des instances YOLOv8-seg.
    Chaque ligne: "c x1 y1 x2 y2 ... xN yN" (coords normalisées [0,1]).
    """
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with lbl_path.open("w") as f:
        for c, poly in instances:
            if poly is None or len(poly) < 3:
                continue
            flat = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
            f.write(f"{int(c)} {flat}\n")


# =========================
# Conversions N<->ABS
# =========================


def seg_n_to_abs(poly_n: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Normalisé -> pixels. poly_n: (N,2) in [0,1], retourne (N,2) en pixels.
    """
    if poly_n.size == 0:
        return poly_n
    poly = poly_n.copy().astype(np.float32)
    poly[:, 0] *= W
    poly[:, 1] *= H
    return poly


def seg_abs_to_n(poly_abs: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Pixels -> normalisé. poly_abs: (N,2) pixels -> (N,2) in [0,1].
    """
    if poly_abs.size == 0:
        return poly_abs
    poly = poly_abs.copy().astype(np.float32)
    poly[:, 0] = np.clip(poly[:, 0] / max(W, 1e-6), 0.0, 1.0)
    poly[:, 1] = np.clip(poly[:, 1] / max(H, 1e-6), 0.0, 1.0)
    return poly


# =========================
# Utilitaires polygones
# =========================


def polygon_area(poly: np.ndarray) -> float:
    """
    Aire signée (positive si sommets dans l'ordre anti-horaire).
    poly: (N,2)
    """
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """
    Force l'ordre anti-horaire (utile mais pas obligatoire).
    """
    if polygon_area(poly) < 0:
        return poly[::-1].copy()
    return poly


def remove_duplicate_consecutive_points(
    poly: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Retire les points consécutifs identiques (numériquement).
    """
    if len(poly) <= 1:
        return poly
    keep = [True]
    for i in range(1, len(poly)):
        if np.linalg.norm(poly[i] - poly[i - 1]) <= eps:
            keep.append(False)
        else:
            keep.append(True)
    out = poly[np.array(keep)]
    # fermer/extrême identique?
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) <= eps:
        out = out[:-1]
    return out


# -------------------------
# Sutherland–Hodgman clip
# -------------------------


def _inside(
    p: np.ndarray, edge: str, x1: float, y1: float, x2: float, y2: float
) -> bool:
    """
    Edge in { 'left', 'right', 'top', 'bottom' }
    Rect: [x1,y1] - [x2,y2]
    """
    if edge == "left":
        return p[0] >= x1
    if edge == "right":
        return p[0] <= x2
    if edge == "top":
        return p[1] >= y1
    if edge == "bottom":
        return p[1] <= y2
    raise ValueError(edge)


def _intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    edge: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """
    Intersection segment p1-p2 avec la droite du bord 'edge' du rectangle.
    """
    x3, y3 = p1
    x4, y4 = p2
    if edge == "left":
        x = x1
        t = (x - x3) / (x4 - x3 + 1e-12)
        y = y3 + t * (y4 - y3)
        return np.array([x, y], dtype=np.float32)
    if edge == "right":
        x = x2
        t = (x - x3) / (x4 - x3 + 1e-12)
        y = y3 + t * (y4 - y3)
        return np.array([x, y], dtype=np.float32)
    if edge == "top":
        y = y1
        t = (y - y3) / (y4 - y3 + 1e-12)
        x = x3 + t * (x4 - x3)
        return np.array([x, y], dtype=np.float32)
    if edge == "bottom":
        y = y2
        t = (y - y3) / (y4 - y3 + 1e-12)
        x = x3 + t * (x4 - x3)
        return np.array([x, y], dtype=np.float32)
    raise ValueError(edge)


def clip_polygon_to_rect(
    poly: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """
    Sutherland–Hodgman: clip 'poly' par le rectangle [x1,y1,x2,y2] (inclusif).
    Retourne un polygone (N,2) en pixels, ou array shape (0,2) si vide.
    """
    if poly is None or len(poly) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    subject = poly.astype(np.float32)
    for edge in ["left", "right", "top", "bottom"]:
        if subject.shape[0] == 0:
            break
        output = []
        S = subject[-1]
        for E in subject:
            if _inside(E, edge, x1, y1, x2, y2):
                if _inside(S, edge, x1, y1, x2, y2):
                    output.append(E)
                else:
                    output.append(_intersect(S, E, edge, x1, y1, x2, y2))
                    output.append(E)
            else:
                if _inside(S, edge, x1, y1, x2, y2):
                    output.append(_intersect(S, E, edge, x1, y1, x2, y2))
            S = E
        subject = (
            np.array(output, dtype=np.float32)
            if output
            else np.zeros((0, 2), dtype=np.float32)
        )

    subject = remove_duplicate_consecutive_points(subject)
    if len(subject) >= 3:
        subject = ensure_ccw(subject)
    return subject


# ==================================
# Tuilage images + labels (segmentation)
# ==================================


def generate_tiles_for_image_seg(
    img_path: Path,
    lbl_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile_size: int = 1024,
    overlap: int = 0,
    min_poly_area: float = 4.0,
    min_poly_points: int = 3,
) -> List[Tuple[Path, Path]]:
    """
    Génère des tuiles pour une image + labels YOLOv8-seg (polygones normalisés).
    Retour: liste des (img_out, lbl_out) créés.
    - min_poly_area: aire min du polygone (en pixels^2) après clip.
    - min_poly_points: nb min de points après clip.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return []

    H, W = img.shape[:2]
    instances = load_yolo_seg_labels(lbl_path)
    # convertit les polygones en pixels
    instances_abs = [(c, seg_n_to_abs(poly, W, H)) for (c, poly) in instances]

    stride = max(1, tile_size - overlap)
    tiles = []

    xs = list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0]

    base = img_path.stem

    # Skip si des tuiles existent déjà pour cette image (même tile_size)
    existing_tiles = list(out_img_dir.glob(f"{base}_x*_y*_ts{tile_size}.jpg"))
    if existing_tiles:
        print(
            f"[tiling] Skip '{img_path.name}': tuiles déjà présentes dans {out_img_dir}."
        )
        return []

    for ty in ys:
        for tx in xs:
            x2 = min(tx + tile_size, W)
            y2 = min(ty + tile_size, H)
            x1 = x2 - tile_size if (x2 - tx) < tile_size else tx
            y1 = y2 - tile_size if (y2 - ty) < tile_size else ty

            tile = img[y1 : y1 + tile_size, x1 : x1 + tile_size].copy()

            # Clip/translate polygones vers le repère de la tuile
            t_instances_abs = []
            for c, poly_abs in instances_abs:
                if poly_abs is None or len(poly_abs) < 3:
                    continue
                # clip par le rectangle de la tuile en coordonnées globales
                clipped = clip_polygon_to_rect(
                    poly_abs, x1, y1, x1 + tile_size, y1 + tile_size
                )
                if clipped.shape[0] < min_poly_points:
                    continue
                # décalage dans le repère local de la tuile
                clipped[:, 0] -= x1
                clipped[:, 1] -= y1

                # filtre sur aire
                area = abs(polygon_area(clipped))
                if area < min_poly_area:
                    continue

                t_instances_abs.append((c, clipped))

            # Convertit vers normalisé (par rapport à la tuile)
            t_instances_n = [
                (c, seg_abs_to_n(poly, tile_size, tile_size))
                for (c, poly) in t_instances_abs
            ]

            out_name = f"{base}_x{x1}_y{y1}_ts{tile_size}.jpg"
            out_img_path = out_img_dir / out_name
            out_lbl_path = out_lbl_dir / (Path(out_name).stem + ".txt")

            out_img_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), tile)
            save_yolo_seg_labels(out_lbl_path, t_instances_n)
            tiles.append((out_img_path, out_lbl_path))

    return tiles


def preprocess_dataset_with_tiles_seg(
    data_root: Path,
    split_names: List[str],
    tile_size: int,
    overlap: int,
    out_root: Path,
    min_poly_area: float = 4.0,
    min_poly_points: int = 3,
):
    """
    Suppose une structure standard YOLO:
      data_root/images/{train,val,test}
      data_root/labels/{train,val,test}
    Crée un dataset tilé (images + labels de segmentation) sous:
      out_root/images/{train,val,test}
      out_root/labels/{train,val,test}
    """
    for split in split_names:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split

        if not img_dir.exists():
            print(f"[tiling] Skip split '{split}': {img_dir} introuvable.")
            continue

        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
            img_paths.extend(img_dir.rglob(ext))

        print(f"[tiling] Split '{split}': {len(img_paths)} images à tuiler...")
        for i, img_path in enumerate(sorted(img_paths)):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            generate_tiles_for_image_seg(
                img_path=img_path,
                lbl_path=lbl_path,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
                tile_size=tile_size,
                overlap=overlap,
                min_poly_area=min_poly_area,
                min_poly_points=min_poly_points,
            )
            if (i + 1) % 50 == 0:
                print(f"[tiling] {split}: {i+1}/{len(img_paths)} images traitées")
        print(f"[tiling] Split '{split}' terminé.")


# =========
# Entrée CL
# =========
if __name__ == "__main__":
    from .settings import DATASET_RAW_DIR, DATASET_TILED_DIR

    preprocess_dataset_with_tiles_seg(
        data_root=DATASET_RAW_DIR,
        split_names=["train", "val"],
        tile_size=1024,
        overlap=256,
        out_root=DATASET_TILED_DIR,
        min_poly_area=4.0,
        min_poly_points=3,
    )
