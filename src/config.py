import os
import torch


# ==========================================
# CAMINHOS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


IMAGE_DIR = os.path.join(
    DATA_DIR,
    "ISIC_reduzido",
    "images"
)


TRAIN_CSV = os.path.join(
    DATA_DIR,
    "ISIC_reduzido",
    "metadata.csv"
)


# ==========================================
# MODELO
# ==========================================

MODEL_NAME = "efficientnet_b0"

NUM_CLASSES = 1


# ==========================================
# IMAGEM
# ==========================================

IMAGE_SIZE = 224


# ==========================================
# TREINO
# ==========================================

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.0001

SEED = 42


# ==========================================
# HARDWARE
# ==========================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Windows + CPU
NUM_WORKERS = 0


# ==========================================
# CHECKPOINT
# ==========================================

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)


BEST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# ==========================================
# SAÍDAS
# ==========================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)