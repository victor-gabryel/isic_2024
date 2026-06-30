import os
import torch
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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


    print("Avaliando...")


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


    # transformar probabilidades em classes

    pred_classes = [
        1 if p >= 0.5 else 0
        for p in predictions
    ]


    auc = roc_auc_score(
        targets,
        predictions
    )


    acc = accuracy_score(
        targets,
        pred_classes
    )


    precision = precision_score(
        targets,
        pred_classes
    )


    recall = recall_score(
        targets,
        pred_classes
    )


    f1 = f1_score(
        targets,
        pred_classes
    )


    cm = confusion_matrix(
        targets,
        pred_classes
    )


    print("======================")
    print("RESULTADOS")
    print("======================")

    print(f"AUC: {auc:.4f}")

    print(f"Accuracy: {acc:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1-score: {f1:.4f}")


    print()

    print("Matriz de confusão:")

    print(cm)



if __name__ == "__main__":

    main()