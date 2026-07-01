import torch
import numpy as np
import pandas as pd
import os
import json

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

from src.config import *
from src.dataset import load_dataset, create_loaders
from src.model import create_model

OUTPUT="outputs"


os.makedirs(
OUTPUT,
exist_ok=True
)

THRESHOLD=0.91


# Gerando gráficos do dataset original e do dataset reduzido para análise visual da distribuição das classes.
def dataset_graphs(df):
    
    plt.figure(figsize=(6,4))
    
    plt.bar(
    ["Benigno","Maligno"],
    [217183,294]
    )

    plt.title("Dataset Original ISIC 2024")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/dataset_original.png",
        dpi=200
    )

    plt.close()

    plt.figure(figsize=(6,4))

    plt.bar(
        [
            "Benigno",
            "Maligno"
        ],
        [
            len(df[df.malignant==0]),
            len(df[df.malignant==1])
        ]
    )

    plt.title("Dataset Reduzido")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/dataset_reduzido.png",
        dpi=200
    )

    plt.close()


# Loss e ROC-AUC durante o treinamento, para análise visual do desempenho do modelo ao longo das épocas
def training_graphs():

    with open(
        "outputs/history.json"
    ) as f:
        h=json.load(f)

    plt.figure(figsize=(6,4))

    plt.plot(
        h["epoch"],
        h["train_loss"],
        label="Treino"
    )

    plt.plot(
        h["epoch"],
        h["val_loss"],
        label="Validação"
    )

    plt.title("Evolução da Loss")

    plt.legend()

    plt.grid()

    plt.savefig(
        f"{OUTPUT}/loss.png",
        dpi=200
    )

    plt.close()

    plt.figure(figsize=(6,4))

    plt.plot(
        h["epoch"],
        h["train_auc"],
        label="Treino"
    )

    plt.plot(
        h["epoch"],
        h["val_auc"],
        label="Validação"
    )

    plt.title("Evolução ROC-AUC")

    plt.legend()

    plt.grid()

    plt.savefig(
        f"{OUTPUT}/roc_evolution.png",
        dpi=200
    )
    plt.close()


# Avaliação final do modelo, com geração de matriz de confusão e curva ROC para análise visual do desempenho do modelo.
def main():

    df=load_dataset()

    dataset_graphs(df)

    training_graphs()

    _,loader=create_loaders(df)

    model=create_model(DEVICE)

    checkpoint=torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    preds=[]

    labels=[]

    with torch.no_grad():

        for images,y in tqdm(loader):

            images=images.to(DEVICE)
            out=model(images)
            prob=torch.sigmoid(out)
            
            preds.extend(
                prob.cpu().numpy()
            )

            labels.extend(
                y.numpy()
            )

    preds=np.array(preds)

    labels=np.array(labels)

    pred_class=(
        preds>=THRESHOLD
    ).astype(int)

    auc=roc_auc_score(
        labels,
        preds
    )

    print(
        "ROC-AUC:",
        auc
    )

    cm=confusion_matrix(
        labels,
        pred_class
    )

    plt.figure(figsize=(5,4))

    plt.imshow(cm)

    plt.title(
        "Matriz de Confusão"
    )


    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center"
            )

    plt.savefig(
        f"{OUTPUT}/confusion_matrix.png",
        dpi=200
    )

    plt.close()

    fpr,tpr,_=roc_curve(
        labels,
        preds
    )

    plt.figure(figsize=(6,4))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={auc:.3f}"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        f"{OUTPUT}/roc_curve.png",
        dpi=200
    )

    plt.close()

    pd.DataFrame({
        "real":labels,
        "probabilidade":preds,
        "predicao":pred_class
    }).to_csv(
        "outputs/final_predictions.csv",
        index=False
    )

    print("Avaliação finalizada")

if __name__=="__main__":
    main()