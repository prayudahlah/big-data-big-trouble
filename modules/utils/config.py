from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
EXPERIMENTS = PROJECT_ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 224
NUM_CLASSES = 3
CLASS_LABELS = ["0_Recyclable", "1_Electronic", "2_Organic"]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
