# Detecção de Lesões Malignas - ISIC 2024

## Sobre o projeto

Este projeto utiliza Deep Learning para classificação de imagens dermatológicas com objetivo de identificar lesões potencialmente malignas.

Foi utilizado o dataset ISIC 2024 e um modelo de visão computacional baseado em EfficientNet-B0.


## Tecnologias utilizadas

- Python
- PyTorch
- Torchvision
- Timm
- Scikit-learn
- Pandas
- Matplotlib


## Modelo

Arquitetura:

EfficientNet-B0


Entrada:

Imagens dermatológicas 224x224


Saída:

Classificação binária:

0 - Benigno

1 - Maligno


## Dataset

Dataset reduzido utilizado:

Total:
10294 imagens


Distribuição:

Benignas:
10000


Malignas:
294


Devido ao desbalanceamento foi utilizado:

BCEWithLogitsLoss com peso de classe


## Treinamento

Configuração:

Épocas máximas:
20


Otimizador:

AdamW


Learning rate:

0.0001


Batch size:

8


Critério:

BCEWithLogitsLoss


## Resultados


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


## Análise

O modelo apresentou excelente capacidade de separação das classes através da métrica ROC-AUC.

O Recall elevado demonstra que o modelo conseguiu identificar grande parte das lesões malignas.

A precisão apresentou redução devido ao desbalanceamento das classes, gerando falsos positivos.


## Arquivos gerados


checkpoints/

best_model.pth

Modelo treinado


outputs/

roc_curve.png

Curva ROC


confusion_matrix.png

Matriz de confusão


results.txt

Relatório das métricas


## Como executar


Instalar dependências:

```bash
pip install -r requirements.txt