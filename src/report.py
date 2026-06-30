import os


REPORT_PATH = "outputs/results.txt"


def main():

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    texto = """

=====================================
Resultados ISIC 2024
=====================================


Modelo:
EfficientNet-B0


Dataset:
ISIC reduzido


Quantidade de imagens:
10294


Distribuição:

Benignas:
10000

Malignas:
294


-------------------------------------

Métricas:

ROC-AUC:
0.9605


Accuracy:
0.8771


Precision:
0.1767


Recall:
0.8983


F1-score:
0.2953


-------------------------------------

Matriz de confusão:

[[1753 247]
 [6    53]]


-------------------------------------

Interpretação:

O modelo apresentou alta capacidade
de separação entre lesões benignas e
malignas, obtendo ROC-AUC de 0.9605.

A sensibilidade (Recall) alcançou
89.83%, indicando boa capacidade de
identificação das lesões malignas.

A precisão apresentou valor reduzido
devido ao desbalanceamento entre as
classes, resultando em maior quantidade
de falsos positivos.

Em aplicações médicas, priorizar a
redução de falsos negativos é importante,
portanto o alto Recall apresentado é
um resultado relevante.


=====================================

"""


    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(texto)


    print(
        "Relatório criado:"
    )

    print(
        REPORT_PATH
    )



if __name__ == "__main__":

    main()