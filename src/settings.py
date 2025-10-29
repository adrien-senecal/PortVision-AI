from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
# Check if the environment variable is set
if os.getenv("DOTA_DIR") is None:
    DOTA_DIR = Path("/data/")
else:
    DOTA_DIR = Path(os.getenv("DOTA_DIR"))

DATASET_RAW_DIR = DOTA_DIR / "dataset_raw"
DATASET_TILED_DIR = DOTA_DIR / "dataset_tiled"
MODEL_DIR = DOTA_DIR / "model"
PREDICTIONS_DIR = DOTA_DIR / "predictions"
