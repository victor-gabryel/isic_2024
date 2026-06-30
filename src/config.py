import torch


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_NAME = "efficientnet_b0"


IMAGE_SIZE = 224


BATCH_SIZE = 32


EPOCHS = 20


LEARNING_RATE = 1e-4


BEST_MODEL = "checkpoints/best_model.pth"