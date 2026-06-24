# src/predict.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch

from config import *
from dataset import (
    load_dataset,
    val_transform
)
from model import create_model


# ==========================
# CARREGAR MODELO
# ==========================

model = create_model().to(DEVICE)

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# ==========================
# PREDIÇÃO
# ==========================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_np = np.array(image)

    image_tensor = val_transform(
        image=image_np
    )["image"]

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        ).item()

    threshold = 0.5

    label = (
        "Maligno"
        if probability >= threshold
        else "Benigno"
    )

    return probability, label


# ==========================
# TESTE MANUAL
# ==========================

def test_random_image():

    df = load_dataset()

    sample = df.sample(1)

    image_path = sample[
        "image_path"
    ].iloc[0]

    real_class = int(
        sample["malignant"].iloc[0]
    )

    probability, prediction = predict_image(
        image_path
    )

    real_label = (
        "Maligno"
        if real_class == 1
        else "Benigno"
    )

    predicted_class = (
        1
        if prediction == "Maligno"
        else 0
    )

    correct = (
        predicted_class
        == real_class
    )

    image = Image.open(
        image_path
    )

    plt.figure(
        figsize=(6,6)
    )

    plt.imshow(image)

    plt.axis("off")

    plt.title(
        f"Real: {real_label}\n"
        f"Previsto: {prediction}\n"
        f"Probabilidade: {probability:.4f}\n"
        f"{'✅ ACERTOU' if correct else '❌ ERROU'}"
    )

    plt.show()

    print(
        "Imagem:",
        os.path.basename(image_path)
    )

    print(
        "Classe real:",
        real_label
    )

    print(
        "Classe prevista:",
        prediction
    )

    print(
        "Probabilidade:",
        probability
    )


# ==========================
# EXECUÇÃO
# ==========================

if __name__ == "__main__":

    test_random_image()