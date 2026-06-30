import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


from src.config import *



OUTPUT="outputs/final"



import os

os.makedirs(
    OUTPUT,
    exist_ok=True
)



THRESHOLD=0.94




def main():


    print("====================")
    print("RELATÓRIO FINAL")
    print("====================")



    df=pd.read_csv(
        "outputs/final_predictions.csv"
    )



    y_true=df["real"]


    y_prob=df["probabilidade"]



    y_pred=(
        y_prob>=THRESHOLD
    ).astype(int)



    auc=roc_auc_score(
        y_true,
        y_prob
    )



    acc=accuracy_score(
        y_true,
        y_pred
    )


    precision=precision_score(
        y_true,
        y_pred
    )


    recall=recall_score(
        y_true,
        y_pred
    )


    f1=f1_score(
        y_true,
        y_pred
    )



    cm=confusion_matrix(
        y_true,
        y_pred
    )




    print(f"""

====================

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


Threshold:
{THRESHOLD}



Matriz:

{cm}

====================

""")





    with open(
        f"{OUTPUT}/report.txt",
        "w"
    ) as f:


        f.write(

f"""
ISIC 2024

Threshold:
{THRESHOLD}


AUC:
{auc}


Accuracy:
{acc}


Precision:
{precision}


Recall:
{recall}


F1:
{f1}



Confusion Matrix:

{cm}


"""

        )





    # matriz

    plt.figure(
        figsize=(5,4)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )


    plt.xlabel(
        "Predito"
    )

    plt.ylabel(
        "Real"
    )


    plt.title(
        "Matriz de Confusão"
    )


    plt.savefig(
        f"{OUTPUT}/confusion_matrix.png"
    )

    plt.close()





    # ROC


    fpr,tpr,_=roc_curve(
        y_true,
        y_prob
    )



    plt.figure(
        figsize=(6,5)
    )


    plt.plot(
        fpr,
        tpr,
        label=f"AUC={auc:.4f}"
    )


    plt.plot(
        [0,1],
        [0,1],
        "--"
    )


    plt.legend()


    plt.xlabel(
        "False Positive Rate"
    )


    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "ROC Curve"
    )


    plt.savefig(
        f"{OUTPUT}/roc_curve.png"
    )


    plt.close()





    # distribuição


    plt.figure(
        figsize=(7,5)
    )


    plt.hist(
        y_prob[y_true==0],
        bins=30,
        alpha=0.7,
        label="Benigno"
    )


    plt.hist(
        y_prob[y_true==1],
        bins=30,
        alpha=0.7,
        label="Maligno"
    )


    plt.legend()


    plt.title(
        "Distribuição das probabilidades"
    )


    plt.xlabel(
        "Probabilidade"
    )


    plt.ylabel(
        "Quantidade"
    )


    plt.savefig(
        f"{OUTPUT}/probability_distribution.png"
    )


    plt.close()



    print("Relatório criado!")




if __name__=="__main__":

    main()