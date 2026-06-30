import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns


from src.config import *
from src.dataset import load_dataset, create_loaders
from src.model import create_model



THRESHOLD = 0.94



def main():


    print("====================")
    print("Avaliação Final ISIC")
    print("====================")


    df = load_dataset()


    _, val_loader = create_loaders(df)



    model=create_model(
        DEVICE
    )


    checkpoint=torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    model.to(DEVICE)

    model.eval()



    preds=[]
    labels=[]



    with torch.no_grad():


        for images,y in tqdm(val_loader):


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



    pred_class=(preds>=THRESHOLD).astype(int)



    auc=roc_auc_score(
        labels,
        preds
    )


    acc=accuracy_score(
        labels,
        pred_class
    )


    precision=precision_score(
        labels,
        pred_class
    )


    recall=recall_score(
        labels,
        pred_class
    )


    f1=f1_score(
        labels,
        pred_class
    )


    cm=confusion_matrix(
        labels,
        pred_class
    )



    print(f"""

Threshold:
{THRESHOLD}


AUC:
{auc:.4f}


Accuracy:
{acc:.4f}


Precision:
{precision:.4f}


Recall:
{recall:.4f}


F1:
{f1:.4f}



Matriz:

{cm}


""")


    # matriz

    plt.figure(figsize=(5,4))


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )


    plt.xlabel("Predito")
    plt.ylabel("Real")


    plt.savefig(
        "outputs/confusion_final.png"
    )


    plt.close()



    # ROC


    fpr,tpr,_=roc_curve(
        labels,
        preds
    )


    plt.figure(figsize=(6,5))


    plt.plot(
        fpr,
        tpr,
        label=f"AUC={auc:.3f}"
    )


    plt.legend()


    plt.xlabel(
        "False Positive"
    )


    plt.ylabel(
        "True Positive"
    )


    plt.savefig(
        "outputs/roc_final.png"
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


    print("Finalizado!")




if __name__=="__main__":

    main()