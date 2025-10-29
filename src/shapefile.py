import argparse
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Point, Polygon

from .settings import PREDICTIONS_DIR

# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert CSV detections to shapefile using georeferenced TIFF."
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file path with detections (x1,y1,x2,y2,conf,class). Relative to PREDICTIONS_DIR or absolute path.",
    )
    parser.add_argument(
        "--tif",
        type=str,
        required=True,
        help="Georeferenced TIFF file path. Relative to PREDICTIONS_DIR or absolute path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output shapefile path (.shp or .gpkg). Relative to PREDICTIONS_DIR or absolute path.",
    )
    parser.add_argument(
        "--conf_min",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (default: 0.0).",
    )
    parser.add_argument(
        "--class",
        type=int,
        default=None,
        dest="keep_class",
        help="Filter to specific class ID only (e.g., 9). If omitted, keep all classes.",
    )
    parser.add_argument(
        "--centroids",
        action="store_true",
        help="Also create a centroids shapefile alongside the bounding boxes.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------------- #
# Conversion function
# ----------------------------------------------------------------------------- #


def csv_to_shapefile(
    csv_path: Path,
    tif_path: Path,
    output_path: Path,
    conf_min: float = 0.0,
    keep_class: Optional[int] = None,
    create_centroids: bool = False,
) -> None:
    """
    Convert CSV detections to shapefile using georeferenced TIFF.

    Args:
        csv_path: Path to CSV file with columns x1,y1,x2,y2,conf,class
        tif_path: Path to georeferenced TIFF file
        output_path: Output shapefile path (.shp or .gpkg)
        conf_min: Minimum confidence threshold
        keep_class: If provided, filter to this class ID only
        create_centroids: If True, also create a centroids shapefile
    """
    # Resolve paths
    csv_path = (
        (PREDICTIONS_DIR / csv_path)
        if not Path(csv_path).is_absolute()
        else Path(csv_path)
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    tif_path = (
        (PREDICTIONS_DIR / tif_path)
        if not Path(tif_path).is_absolute()
        else Path(tif_path)
    )
    if not tif_path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {tif_path}")

    output_path = (
        (PREDICTIONS_DIR / output_path)
        if not Path(output_path).is_absolute()
        else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Optional filters
    if conf_min is not None and conf_min > 0.0:
        initial_count = len(df)
        df = df[df["conf"] >= conf_min]
        logging.info(
            "Filtered by confidence >= %.3f: %d -> %d detections",
            conf_min,
            initial_count,
            len(df),
        )
    if keep_class is not None:
        initial_count = len(df)
        df = df[df["class"] == keep_class]
        logging.info(
            "Filtered by class == %d: %d -> %d detections",
            keep_class,
            initial_count,
            len(df),
        )

    if len(df) == 0:
        logging.warning("No detections remaining after filtering.")
        return

    # Open raster to get transform + CRS
    logging.info("Reading georeference from TIFF: %s", tif_path)
    with rasterio.open(tif_path) as src:
        transform: Affine = src.transform  # pixel,line -> map
        crs = src.crs

    logging.info("CRS: %s", crs)

    # Helper: pixel/line to map coords using the raster's affine transform
    def px_to_map(x_pix, y_pix):
        Xmap, Ymap = transform * (x_pix, y_pix)
        return Xmap, Ymap

    # Build geometries
    logging.info("Building geometries for %d detections...", len(df))
    poly_geoms = []
    cent_geoms = []
    for _, r in df.iterrows():
        x1, y1, x2, y2 = float(r.x1), float(r.y1), float(r.x2), float(r.y2)
        # Ensure x1<x2, y1<y2 regardless of order
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # Pixel corners (top-left, top-right, bottom-right, bottom-left)
        tl = px_to_map(xmin, ymin)
        tr = px_to_map(xmax, ymin)
        br = px_to_map(xmax, ymax)
        bl = px_to_map(xmin, ymax)

        poly = Polygon([tl, tr, br, bl, tl])
        poly_geoms.append(poly)

        # centroid in pixel coords -> map coords
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        cent_geoms.append(Point(*px_to_map(cx, cy)))

    # Build GeoDataFrames
    gdf_poly = gpd.GeoDataFrame(df.copy(), geometry=poly_geoms, crs=crs)
    gdf_cent = gpd.GeoDataFrame(df.copy(), geometry=cent_geoms, crs=crs)

    # Write outputs
    logging.info("Writing shapefile: %s", output_path)
    gdf_poly.to_file(output_path)

    if create_centroids:
        cent_out = output_path.with_name(
            output_path.stem + "_centroids" + output_path.suffix
        )
        logging.info("Writing centroids shapefile: %s", cent_out)
        gdf_cent.to_file(cent_out)

    logging.info(
        "Conversion complete. Wrote %d features to: %s", len(gdf_poly), output_path
    )
    if create_centroids:
        logging.info("Wrote %d centroids to: %s", len(gdf_cent), cent_out)


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
        csv_to_shapefile(
            csv_path=Path(args.csv),
            tif_path=Path(args.tif),
            output_path=Path(args.output),
            conf_min=args.conf_min,
            keep_class=args.keep_class,
            create_centroids=args.centroids,
        )

    except Exception as exc:
        logging.exception("Error during conversion: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
