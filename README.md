# PortVision-AI

Détection d’objets sur imagerie aérienne/satellite avec YOLOv11, tuilage (sliding window) pour très grandes images, et export géospatial des résultats (Shapefile/GeoPackage).

## Objectifs

Mettre en place un pipeline reproductible pour:

- Mettre en place l’architecture d’un projet Python de vision par ordinateur structuré et maintenable
- Entraîner un modèle YOLO sur DOTA-v2 (ou un dataset au format YOLO)
- Inférer sur de très grandes images via tuilage
- Convertir des détections en données géoréférencées (SHP/GPKG) à partir d’un GeoTIFF

## Introduction

PortVision-AI s’appuie sur la librairie Ultralytics pour entraîner et utiliser des modèles YOLOv11. Il inclut:

- des utilitaires de préparation de données par tuilage pour l’entraînement
- un script d’inférence capable de traiter des images larges avec tuilage + fusion des détections
- un convertisseur CSV → Shapefile/GPKG basé sur le GeoTIFF d’origine pour replacer les détections dans un système de coordonnées cartographiques

Le jeu de classes par défaut correspond à DOTA-v2 (18 classes, cf. `src/class_names.py`).

---

## Structure du projet

```text
PortVision-AI/
  data/
    dotav2.yaml                 # Exemple de YAML dataset YOLO
    raw/                        # Exemples d’images brutes (démo)
  src/
    class_names.py              # Noms de classes DOTA-v2
    settings.py                 # Répertoires pilotés par env var DOTA_DIR
    tiling_utils.py             # Tuilage (images + labels YOLO-seg) pour entraînement
    train.py                    # Entraînement YOLOv11 + pipeline de tuilage
    predict.py                  # Inférence single-shot ou par tuilage + NMS
    shapefile.py                # Conversion CSV → Shapefile/GPKG via GeoTIFF
  notebooks/                    # Modèles pré-entraînés (exemples) et notebooks
  requirements.txt              # Dépendances Python
  Dockerfile                    # Image CUDA + Ultralytics (optionnel)
  README.md
```

---

## Installation (environnement Python)

- Pré-requis: Python ≥ 3.10
- GPU NVIDIA recommandé (CUDA) pour l’entraînement.

### Avec uv (recommandé)

```powershell
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### Sans uv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Vérifiez ensuite que `ultralytics` se lance correctement:

```powershell
python -c "import ultralytics, torch; print('OK', ultralytics.__version__)"
```

---

## Configuration et variables d’environnement

Le projet s’appuie sur la variable `DOTA_DIR` pour définir les dossiers de travail (cf. `src/settings.py`). À défaut, le chemin par défaut est `/data/`.

Sous Windows, créez un fichier `.env` à la racine du projet avec par exemple:

```text
DOTA_DIR=D:/PortVision-Data
```

Arborescence attendue dans `DOTA_DIR` (créée au besoin):

- `dataset_raw/` jeu de données YOLO source (images/labels)
- `dataset_tiled/` jeu de données tuilé (généré)
- `model/` sorties d’entraînement (runs, checkpoints)
- `predictions/` résultats d’inférence (images, CSV, exports)

---

## Données attendues (format YOLO)

`dataset_raw/` doit suivre la convention YOLO:

```text
dataset_raw/
  images/
    train/  val/  test/
  labels/
    train/  val/  test/   # fichiers .txt YOLO (bbox ou seg)
```

- Un exemple de YAML est fourni: `data/dotav2.yaml`. Lors du tuilage pour l’entraînement, un YAML dérivé est généré automatiquement vers `model/dotav2_<suffix>.yaml`.
- Les classes DOTA-v2 sont exposées dans `src/class_names.py`.
- Le dataset est disponible sur Hugging Face : https://huggingface.co/datasets/satellite-image-deep-learning/DOTAv2/blob/main/DOTAv2.zip

---

## Entraînement

Commande type (tuilage activé par défaut):

```powershell
python -m src.train --epochs 10 --batch-size 8 --image-size 1024 \
  --model yolo11n.pt --device 0 \
  --tile-size 1024 --tile-overlap 256 \
  --tiled-dataset-suffix tiles1024
```

Options utiles:

- `--tiling/--no-tiling` active/désactive le prétraitement en tuiles
- `--use-premade-tiles` utilise un dataset déjà tuilé (n’en régénère pas)
- `--clear-cuda-cache` vide le cache CUDA avant démarrage

Le run est enregistré dans `model/` et un checkpoint final est sauvegardé sous un nom descriptif `dota_v2_<model>_eXX_bXX_imgXXXX[_tile...]_final.pt`.

---

## Inférence

Inférence sur une image modérée (sans tuilage):

```powershell
python -m src.predict --model dota_v2_yolo11n_e10_b8_img1024_final.pt \
  --image raw/Pleiades_SaintNazaire.jpg --conf 0.25 --iou 0.45 \
  --device 0 --output_dir demo_run --classes 9 10 15
```

- `--output_dir` active l’enregistrement dans `predictions/results/<output_dir>/prediction/`
- Fichiers produits si `--output_dir` est renseigné:
  - `stitched.png`: visualisation avec boîtes
  - `detections.csv`: colonnes `x1,y1,x2,y2,conf,class`

Inférence sur grande image (tuilage + fusion NMS par classe):

```powershell
python -m src.predict --model <checkpoint.pt> --image <image.tif|png|jpg> \
  --tile --tile_size 1024 --overlap 256 --output_dir bigimg_run
```

Filtrage des classes: utilisez `--classes` (liste d’identifiants; par défaut `9 10 15` véhicules).

---

## Export géospatial (CSV → Shapefile/GPKG)

À partir du `detections.csv` et du GeoTIFF source géoréférencé:

```powershell
python -m src.shapefile --csv results/demo_run/prediction/detections.csv \
  --tif raw/Pleiades_SaintNazaire.tif \
  --output results/demo_run/prediction/detections.gpkg \
  --conf_min 0.3 --class 9 --centroids
```

Notes:

- Les coordonnées pixel sont reprojetées en coordonnées carte via la transformée affine du TIFF
- `--output` peut être `.shp` (ESRI Shapefile) ou `.gpkg` (GeoPackage)
- `--centroids` génère en plus un fichier des centroïdes

---

## Docker (optionnel)

Une image basée sur CUDA est fournie pour exécuter Ultralytics avec GPU. Montez un volume pour `/data` correspondant à `DOTA_DIR`.

### Build:

```bash
docker build -t portvisionai:latest .
```

### Entraînement:

```bash
docker run --gpus all --ipc=host -v "F:/DOTA dataset/DOTAv2:/data" -it portvisionai:latest \
  python -m src.train --clear-cuda-cache --use-premade-tiles --epochs 20 --model yolo11n.pt --batch-size 4
```

### Inférence sur l’image Pléiades tuilée:

```bash
docker run --gpus all -v "F:/DOTA dataset/DOTAv2:/data" -it portvisionai:latest \
  python -m src.predict --model dotav2_yolo11m_e20_b4_img1024_tile1024_ov256.pt --image zone1.tiff --output_dir Zone1 --tile --conf 0.05
```

### Export SIG:

```bash
docker run -v "F:/DOTA dataset/DOTAv2:/data" -it portvisionai:latest \
  python -m src.shapefile --csv detections_zone1.csv --tif zone1.tiff --output shapefiles_zone1
```

Notes:

- Remplacez le chemin Windows du volume par le dossier qui contient vos images et sorties (ex: zone1.tiff, detections_zone1.csv).
- Sous Linux/macOS: -v /abs/path/to/data:/data.
- Sans GPU: supprimez --gpus all (l’inférence sera plus lente).
