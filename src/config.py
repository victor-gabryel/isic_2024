from pathlib import Path
import torch
import os

# =====================================
# CONFIGURAÇÕES GERAIS
# =====================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

IMAGE_SIZE = 300

BATCH_SIZE = 8

EPOCHS = 20

LR = 3e-5

WEIGHT_DECAY = 1e-2

POS_WEIGHT = 10.0

PATIENCE = 5

# =====================================
# CAMINHOS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_CSV = "data/ISIC_2024_Permissive_Training_GroundTruth.csv"

IMAGES_PATH = "data/ISIC_2024_Permissive_Training_Input"

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "checkpoint.pth"

BEST_MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.pth"

OUTPUTS_DIR = BASE_DIR / "outputs"

# =====================================
# CRIA PASTAS AUTOMATICAMENTE
# =====================================

os.makedirs(BASE_DIR / "checkpoints", exist_ok=True)
os.makedirs(BASE_DIR / "outputs", exist_ok=True)