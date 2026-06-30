# Classificação de Lesões Cutâneas utilizando Deep Learning

## 1. Visão Geral

Este projeto apresenta o desenvolvimento de um sistema de classificação automática de lesões cutâneas utilizando técnicas de Deep Learning aplicadas a imagens dermatológicas.

O objetivo é realizar a classificação binária entre lesões benignas e malignas utilizando uma rede neural convolucional com aprendizado por transferência (Transfer Learning), buscando auxiliar processos de triagem e análise automatizada de imagens médicas.

O modelo foi desenvolvido utilizando PyTorch e disponibilizado por meio de uma aplicação interativa desenvolvida em Streamlit.

---

# 2. Tecnologias Utilizadas

* Python
* PyTorch
* Torchvision
* timm
* EfficientNet-B0
* OpenCV
* Albumentations
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit

---

# 3. Dataset

Foi utilizado o dataset:

**ISIC 2024 Permissive Training Input**

O dataset original apresenta forte desbalanceamento entre as classes:

* Aproximadamente 217.183 imagens benignas;
* 294 imagens malignas.

Devido ao grande desequilíbrio entre as categorias, foi utilizado um dataset reduzido localizado em:

```
data/ISIC_reduzido/metadata.csv
```

Esse conjunto foi utilizado para treinamento e avaliação do modelo.

Classes utilizadas:

* Benigno
* Maligno

---

# 4. Estrutura do Projeto

```
isic_2024/

│
├── app.py
│
├── requirements.txt
│
├── src/
│   │
│   ├── model.py
│   └── config.py
│
├── data/
│   │
│   └── ISIC_reduzido/
│       │
│       └── metadata.csv
│
├── outputs/
│   │
│   ├── dataset_original.png
│   ├── dataset_reduzido.png
│   ├── roc_curve.png
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── outputs/gradcam/
│
└── models/
    │
    └── best_model.pth
```

---

# 5. Metodologia

O desenvolvimento do sistema foi dividido nas seguintes etapas:

## 5.1 Preparação do Dataset

O arquivo:

```
data/ISIC_reduzido/metadata.csv
```

foi utilizado para organização das imagens e definição das classes.

As imagens foram separadas em:

* benignas;
* malignas.

---

## 5.2 Pré-processamento das Imagens

Antes da entrada no modelo, as imagens passam pelo seguinte pipeline:

* Redimensionamento para 224x224 pixels;
* Conversão para Tensor;
* Normalização utilizando os valores do ImageNet.

Implementado utilizando:

```
torchvision.transforms
```

---

# 6. Experimentos Realizados

## 6.1 Configuração do Modelo

A arquitetura utilizada foi:

```
EfficientNet-B0
```

O modelo foi carregado utilizando a biblioteca:

```
timm
```

com pesos pré-treinados:

```
pretrained=True
```

O código principal está localizado em:

```
src/model.py
```

A arquitetura utilizada:

```
EfficientNet-B0
        |
Extração de características
        |
Dropout(0.4)
        |
Linear(features,1)
        |
Classificação binária
```

---

## 6.2 Estratégia de Treinamento

O treinamento foi realizado utilizando aprendizado por transferência.

A rede inicial utiliza conhecimentos adquiridos no dataset ImageNet e foi adaptada para classificação binária de lesões cutâneas.

---

## 6.3 Data Augmentation

Foram utilizadas técnicas de aumento de dados para aumentar a diversidade das imagens durante o treinamento.

Transformações aplicadas:

* Rotação;
* Flip horizontal;
* Flip vertical;
* Ajuste de brilho;
* Contraste;
* CLAHE;
* Transformações geométricas.

A biblioteca utilizada foi:

```
Albumentations
```

---

## 6.4 Tratamento do Desbalanceamento

Devido à diferença entre a quantidade de imagens benignas e malignas, foram utilizadas estratégias para reduzir o impacto da classe majoritária.

Foram aplicadas:

* redução do conjunto benigno;
* aumento da diversidade através de Data Augmentation;
* ponderação da classe minoritária durante treinamento.

---

# 7. Avaliação Experimental

O modelo foi avaliado utilizando métricas de classificação:

* Accuracy;
* Precision;
* Recall;
* F1-score;
* ROC-AUC;
* Matriz de Confusão.

Resultados obtidos:

| Métrica        | Resultado             |
| -------------- | --------------------- |
| ROC-AUC        | aproximadamente 0.903 |
| Recall maligno | aproximadamente 73%   |
| Acurácia       | aproximadamente 90%   |

---

# 8. Aplicação Streamlit

A aplicação está localizada em:

```
app.py
```

A interface permite:

## Predição Individual

O usuário pode enviar uma imagem e o sistema retorna:

* classe prevista;
* confiança da previsão.

---

## Testes Automáticos

O usuário pode selecionar:

* 10 imagens aleatórias;
* 10 imagens benignas;
* 10 imagens malignas;
* 5 benignas + 5 malignas.

Para cada imagem são exibidos:

* ID da imagem;
* classe real;
* previsão da IA;
* nível de confiança;
* indicação de acerto ou erro.

---

# 9. Visualização dos Resultados

Os gráficos utilizados pela aplicação estão localizados em:

```
outputs/
```

Arquivos:

```
dataset_original.png
```

Distribuição das classes no dataset original.

```
dataset_reduzido.png
```

Distribuição das classes no dataset reduzido.

```
roc_curve.png
```

Curva ROC-AUC do modelo.

```
accuracy.png
```

Evolução da acurácia durante treinamento.

```
loss.png
```

Evolução da função de perda.

```
confusion_matrix.png
```

Matriz de confusão da classificação.

---

# 10. Grad-CAM

Foi implementada uma etapa de interpretabilidade utilizando Grad-CAM.

Os resultados são armazenados em:

```
outputs/gradcam/
```

Essa técnica permite visualizar quais regiões da imagem tiveram maior influência na decisão da rede neural.

---

# 11. Como Executar

Instalar dependências:

```bash
pip install -r requirements.txt
```

Executar a aplicação:

```bash
streamlit run app.py
```

---

# 12. Modelo Treinado

O modelo treinado deve estar localizado em:

```
models/best_model.pth
```

O carregamento do modelo é realizado através de:

```
src/config.py
```

e:

```
src/model.py
```

---

# 13. Autores

Rafael Reis Borges da Silva
Victor Gabryel da Silva

Curso Superior de Tecnologia em Análise e Desenvolvimento de Sistemas

Instituto Federal de Educação, Ciência e Tecnologia de Pernambuco
IFPE - Campus Jaboatão dos Guararapes

---

# 14. Observação

O sistema possui finalidade acadêmica e experimental.

O modelo desenvolvido funciona como ferramenta de apoio baseada em Inteligência Artificial e não substitui avaliação médica especializada.