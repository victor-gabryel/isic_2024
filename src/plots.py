import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.config import *
from src.dataset import create_loaders
from src.model import create_model


def main():

    print("Carregando dataset...")

    df = pd.read_csv(TRAIN_CSV)

    df["image_path"] = df["isic_id"].apply(
        lambda x: os.path.join(
            IMAGE_DIR,
            f"{x}.jpg"
        )
    )


    _, val_loader = create_loaders(df)


    print("Carregando modelo...")


    model = create_model(DEVICE)

    model.load_state_dict(
        torch.load(
            BEST_MODEL,
            map_location=DEVICE
        )
    )


    model.to(DEVICE)

    model.eval()


    predictions = []

    targets = []


    print("Gerando previsões...")


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)


            outputs = model(images)


            probs = torch.sigmoid(outputs)


            predictions.extend(
                probs.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )


    # =========================
    # CURVA ROC
    # =========================


    fpr, tpr, _ = roc_curve(
        targets,
        predictions
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    plt.figure(figsize=(7,5))


    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )


    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )


    plt.xlabel(
        "Falso Positivo"
    )

    plt.ylabel(
        "Verdadeiro Positivo"
    )


    plt.title(
        "Curva ROC - ISIC 2024"
    )


    plt.legend()


    plt.savefig(
        "outputs/roc_curve.png",
        dpi=300
    )


    plt.close()



    # =========================
    # MATRIZ CONFUSÃO
    # =========================


    pred_classes = [
        1 if p >= 0.5 else 0
        for p in predictions
    ]


    cm = confusion_matrix(
        targets,
        pred_classes
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Benigno",
            "Maligno"
        ]
    )


    disp.plot()


    plt.title(
        "Matriz de Confusão - ISIC 2024"
    )


    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300
    )


    plt.close()



    print("Gráficos criados!")


if __name__ == "__main__":

    main()