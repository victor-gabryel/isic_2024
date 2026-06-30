import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

OUTPUT = "outputs/final_report"

os.makedirs(
    OUTPUT,
    exist_ok=True
)

def main():
    print("====================")
    print("GERANDO RESULTADOS FINAIS")
    print("====================")

    csv_path = "outputs/final_predictions.csv"

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            "Arquivo final_predictions.csv não encontrado"
        )

    df = pd.read_csv(csv_path)

    print("\nDados encontrados:")
    print(df.head())

    real = df["real"]
    prob = df["probabilidade"]

    pred = df["predicao"]

    auc = roc_auc_score(
        real,
        prob
    )

    cm = confusion_matrix(
        real,
        pred
    )

    tn,fp,fn,tp = cm.ravel()

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    precision = tp/(tp+fp+1e-8)
    
    recall = tp/(tp+fn+1e-8)

    f1 = (
        2 *
        precision *
        recall /
        (precision+recall+1e-8)
    )


    # CSV Métricas

    metrics = pd.DataFrame({

        "metrica":[
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ],

        "valor":[
            auc,
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics.to_csv(

        f"{OUTPUT}/metricas.csv",

        index=False
    )


    # Texto
    
    with open(
        f"{OUTPUT}/metricas_finais.txt",
        "w"
    ) as f:
        f.write (

            f"""
            MODELO ISIC 2024

            Threshold utilizado: 0.94

            AUC: {auc:.4f}

            Accuracy: {accuracy:.4f}

            Precision: {precision:.4f}

            Recall: {recall:.4f}

            F1: {f1:.4f}

            Matriz Confusão: {cm}
            """
        )


    # Matriz

    plt.figure(
        figsize=(5,4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
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
        f"{OUTPUT}/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # Roc

    fpr,tpr,_ = roc_curve(
        real,
        prob
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

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.savefig(

        f"{OUTPUT}/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # Resumo

    with open(
        f"{OUTPUT}/resumo_modelo.txt",
        "w"
    ) as f:
        f.write(
            """
            Modelo:
            EfficientNet ISIC 2024


            Objetivo:
            Classificação de lesões de pele
            (Maligno x Benigno)

            Threshold:
            0.94

            Método:
            Deep Learning com Transfer Learning

            Interpretabilidade:
            Grad-CAM

            """
        )

    print("\nResultados criados!")

    print(OUTPUT)

if __name__=="__main__":
    main()