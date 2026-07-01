import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from src.config import *
from src.dataset import load_dataset, create_loaders
from src.model import create_model


# Configurações gerais para avaliação do modelo, incluindo o diretório de saída para salvar gráficos e resultados, bem como o limiar (THRESHOLD) utilizado para classificar as predições como malignas ou benignas.
OUTPUT = "outputs"

os.makedirs(
    OUTPUT,
    exist_ok=True
)

THRESHOLD = 0.91


# Gráficos que representam a distribuição das classes no dataset original e no dataset reduzido. A função dataset_graphs recebe um dataframe (df) e gera dois gráficos de barras, um para o dataset original e outro para o dataset reduzido, salvando-os no diretório especificado em OUTPUT.
def dataset_graphs(df):

    # Dataset original

    plt.figure(figsize=(6,4))

    plt.bar(
        ["Benigno","Maligno"],
        [217183,294],
        color=["steelblue","crimson"]
    )

    plt.title("Dataset Original ISIC 2024")
    plt.ylabel("Quantidade")
    plt.grid(axis="y",alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/dataset_original.png",
        dpi=300
    )
    plt.close()

    # Dataset reduzido
    benigno=len(df[df.malignant==0])
    maligno=len(df[df.malignant==1])
    plt.figure(figsize=(6,4))

    plt.bar(
        ["Benigno","Maligno"],
        [benigno,maligno],
        color=["steelblue","crimson"]
    )

    plt.title("Dataset Reduzido")
    plt.ylabel("Quantidade")
    plt.grid(axis="y",alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/dataset_reduzido.png",
        dpi=300
    )

    plt.close()


# Gera gráficos de treinamento, incluindo a evolução da Loss e da ROC-AUC ao longo das épocas. A função training_graphs lê o arquivo history.json gerado durante o treinamento do modelo e cria gráficos de linha para visualizar o desempenho do modelo durante o processo de treinamento.
def training_graphs():

    history_path="outputs/history.json"

    if not os.path.exists(history_path):
        print("history.json não encontrado.")
        return

    with open(history_path,"r") as f:
        history=json.load(f)

    # Loss é uma métrica que mede o quão bem o modelo está se ajustando aos dados de treinamento. A função training_graphs cria um gráfico de linha mostrando a evolução da Loss ao longo das épocas, tanto para os dados de treino quanto para os dados de validação, permitindo avaliar se o modelo está aprendendo de forma eficaz ou se está sofrendo de overfitting ou underfitting.
    plt.figure(figsize=(7,5))
    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        linewidth=2,
        label="Treino"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        linewidth=2,
        label="Validação"
    )

    plt.title("Evolução da Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/loss.png",
        dpi=300
    )

    plt.close()

    # ROC-AUC é uma métrica que avalia a capacidade do modelo em distinguir entre classes (maligno e benigno). A função training_graphs cria um gráfico de linha mostrando a evolução da ROC-AUC ao longo das épocas, tanto para os dados de treino quanto para os dados de validação, permitindo avaliar a performance do modelo na classificação das imagens.
    plt.figure(figsize=(7,5))
    plt.plot(
        history["epoch"],
        history["train_auc"],
        marker="o",
        linewidth=2,
        label="Treino"
    )

    plt.plot(
        history["epoch"],
        history["val_auc"],
        marker="o",
        linewidth=2,
        label="Validação"
    )

    plt.title("Evolução da ROC-AUC")
    plt.xlabel("Época")
    plt.ylabel("ROC-AUC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/roc_auc.png",
        dpi=300
    )

    plt.close()


# Avaliação do modelo é realizada através da função evaluate_model, que carrega o dataset, gera gráficos de distribuição das classes, cria loaders para inferência, carrega o modelo treinado e realiza a inferência nas imagens. Em seguida, calcula métricas de desempenho como ROC-AUC, Accuracy, Precision, Recall e F1-score, além de gerar gráficos representando essas métricas, a curva ROC e a matriz de confusão. Por fim, salva as predições finais em um arquivo CSV e exibe um resumo das métricas calculadas.
def evaluate_model():

    print("="*60)
    print("AVALIAÇÃO DO MODELO")
    print("="*60)

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
    predictions=[]
    probabilities=[]
    labels=[]

    # Inferência é o processo de utilizar o modelo treinado para fazer previsões em novos dados. A função evaluate_model realiza a inferência nas imagens do dataset, calculando as probabilidades de cada imagem pertencer à classe maligno ou benigno, e aplicando um limiar (THRESHOLD) para classificar as predições. As probabilidades, predições e rótulos reais são armazenados em listas para posterior cálculo das métricas de desempenho.
    with torch.no_grad():

        for images, y in tqdm(loader):
            images = images.to(DEVICE)
            output = model(images)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            pred = (probs >= THRESHOLD).astype(int)
            probabilities.extend(probs)
            predictions.extend(pred)
            labels.extend(y.numpy())

    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Métricas de desempenho são calculadas com base nas predições do modelo e nos rótulos reais das imagens. A função evaluate_model utiliza funções da biblioteca sklearn para calcular métricas como ROC-AUC, Accuracy, Precision, Recall e F1-score, que fornecem uma visão abrangente da performance do modelo na classificação das imagens como malignas ou benignas.
    auc = roc_auc_score(
        labels,
        probabilities
    )
    accuracy = accuracy_score(
        labels,
        predictions
    )
    
    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    print()
    print("="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"ROC-AUC    : {auc:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"Threshold  : {THRESHOLD}")
    print("="*60)
    
    
    # Gráfico de métricas é gerado para visualizar o desempenho do modelo em diferentes métricas de avaliação. A função evaluate_model cria um gráfico de barras que exibe os valores de ROC-AUC, Accuracy, Precision, Recall e F1-score, permitindo uma análise rápida e visual da performance do modelo na classificação das imagens.
    metricas = [
        "ROC-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    valores = [
        auc,
        accuracy,
        precision,
        recall,
        f1
    ]

    plt.figure(figsize=(7,5))

    bars = plt.bar(
        metricas,
        valores
    )

    plt.ylim(0,1)
    plt.title("Métricas do Modelo")
    plt.ylabel("Valor")
    plt.grid(axis="y",alpha=0.3)

    for bar, valor in zip(bars, valores):
        plt.text(
            bar.get_x()+bar.get_width()/2,
            valor+0.01,
            f"{valor:.3f}",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/metrics.png",
        dpi=300
    )

    plt.close()

    # Curva ROC é gerada para avaliar a capacidade do modelo em distinguir entre classes (maligno e benigno). A função evaluate_model calcula os valores de False Positive Rate (FPR) e True Positive Rate (TPR) utilizando a função roc_curve da biblioteca sklearn, e cria um gráfico da curva ROC, que é salvo no diretório especificado em OUTPUT.
    fpr, tpr, _ = roc_curve(
        labels,
        probabilities
    )

    plt.figure(figsize=(7,5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--",
        color="gray"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/roc_curve.png",
        dpi=300
    )

    plt.close()

    # Matriz de confusão é gerada para visualizar o desempenho do modelo na classificação das imagens em classes malignas e benignas. A função evaluate_model utiliza a função confusion_matrix da biblioteca sklearn para calcular a matriz de confusão com base nas predições do modelo e nos rótulos reais, e cria um gráfico que exibe os valores da matriz, permitindo uma análise detalhada dos acertos e erros do modelo.
    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(figsize=(6,5))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues
    )

    plt.title("Matriz de Confusão")
    plt.colorbar()
    classes = ["Benigno", "Maligno"]
    tick_marks = np.arange(len(classes))

    plt.xticks(
        tick_marks,
        classes
    )

    plt.yticks(
        tick_marks,
        classes
    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12
            )

    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT}/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Salvar predições finais é realizado para registrar os resultados da inferência do modelo em um arquivo CSV. A função evaluate_model cria um DataFrame contendo os rótulos reais, as probabilidades previstas e as predições finais (maligno ou benigno) para cada imagem, e salva esse DataFrame em um arquivo chamado final_predictions.csv no diretório especificado em OUTPUT, permitindo uma análise posterior dos resultados do modelo.
    resultados = pd.DataFrame({
        "real": labels,
        "probabilidade": probabilities,
        "predicao": predictions
    })

    resultados.to_csv(
        f"{OUTPUT}/final_predictions.csv",
        index=False
    )


    # Resumo final das métricas
    print()
    print("="*60)
    print("Arquivos gerados:")
    print("="*60)
    print("dataset_original.png")
    print("dataset_reduzido.png")
    print("loss.png")
    print("roc_auc.png")
    print("roc_curve.png")
    print("metrics.png")
    print("confusion_matrix.png")
    print("final_predictions.csv")
    print()
    print("Avaliação concluída com sucesso!")


# Função principal
def main():
    evaluate_model()

if __name__ == "__main__":
    main()