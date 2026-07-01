# Classificação de Lesões Cutâneas utilizando Deep Learning

# 1. Visão Geral

Este projeto apresenta o desenvolvimento de um sistema de classificação automática de lesões cutâneas utilizando técnicas de Deep Learning aplicadas a imagens dermatológicas.

O sistema utiliza **Transfer Learning** com a arquitetura **EfficientNet-B0**, implementada em **PyTorch**, para realizar a classificação binária entre lesões **benignas** e **malignas**. Além do treinamento e avaliação do modelo, o projeto disponibiliza uma interface web desenvolvida em **Streamlit**, permitindo realizar inferências em novas imagens e visualizar métricas e gráficos do treinamento.

[![Abrir Aplicação](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://isic2024-6vvmjqgewe2qfioxuyccga.streamlit.app/)

---

# 2. Tecnologias Utilizadas

- Python
- PyTorch
- Torchvision
- timm
- EfficientNet-B0
- Albumentations
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

---

# 3. Dataset

Dataset utilizado:

**ISIC 2024 – Permissive Training Input**

O conjunto original possui aproximadamente:

- 217.183 imagens benignas
- 294 imagens malignas

Para reduzir o desbalanceamento foi criado um subconjunto localizado em:

```text
data/
└── ISIC_reduzido/
    ├── metadata.csv
    └── images/
```

Classes:

- Benigno
- Maligno

---

# 4. Estrutura do Projeto

```text
isic_2024/
│
├── app.py
├── requirements.txt
├── README.md
│
├── checkpoints/
│   └── best_model.pth
│
├── outputs/
│   ├── metrics.png
│   ├── confusion_matrix.png
│   ├── dataset_original.png
│   ├── dataset_reduzido.png
│   ├── final_predictions.csv
│   ├── history.json
│   ├── loss.png
│   ├── roc_curve.png
│   ├── roc_auc.png
│   └── gradcam/
│
├── data/
│   └── ISIC_reduzido/
│       ├── metadata.csv
│       └── images/
│
└── src/
    ├── config.py
    ├── dataset.py
    ├── evaluate.py
    ├── find_threshold.py
    ├── gradcam.py
    ├── model.py
    └── train.py
```

---

# 5. Metodologia

## Pré-processamento

- Resize 224×224
- Conversão para Tensor
- Normalização ImageNet

## Arquitetura

```
EfficientNet-B0
        ↓
Extração de características
        ↓
Dropout (0.4)
        ↓
Linear(features,1)
        ↓
Sigmoid
```

Implementação em:

```text
src/model.py
```

---

# 6. Experimentos

Foram realizados:

- Transfer Learning
- Data Augmentation
- Ajuste de threshold
- Avaliação final
- Grad-CAM

Data augmentation:

- Flip Horizontal
- Flip Vertical
- Rotação
- Ajuste de brilho
- Contraste
- CLAHE
- Transformações geométricas

---

# 7. Treinamento

Arquivo:

```text
src/train.py
```

Durante o treinamento são registrados:

- Loss de treino
- Loss de validação
- ROC-AUC de treino
- ROC-AUC de validação

Os resultados são armazenados em:

```text
outputs/history.json
```

O melhor modelo é salvo automaticamente em:

```text
checkpoints/best_model.pth
```

---

# 8. Avaliação

Arquivo:

```text
src/evaluate.py
```

São calculadas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de Confusão

Também são gerados:

- accuracy.png
- loss.png
- roc_curve.png
- roc_evolution.png
- confusion_matrix.png
- dataset_original.png
- dataset_reduzido.png

---

# 9. Aplicação Web

Arquivo:

```text
app.py
```

Funcionalidades:

- Upload de imagem
- Classificação Benigno/Maligno
- Confiança da predição
- Visualização das métricas
- Visualização dos gráficos
- Testes automáticos com imagens do dataset

---

# 10. Grad-CAM

Arquivo:

```text
src/gradcam.py
```

As imagens geradas são armazenadas em:

```text
outputs/gradcam/
```

---

# 11. Como Executar

Instalar dependências:

```bash
pip install -r requirements.txt
```

Treinar:

```bash
python -m src.train
```

Encontrar o melhor threshold:

```bash
python -m src.find_threshold
```

Gerar métricas e gráficos:

```bash
python -m src.evaluate
```

Gerar Grad-CAM:

```bash
python -m src.gradcam
```

Executar a aplicação:

```bash
streamlit run app.py
```

---

# 12. Resultados

Modelo:

- EfficientNet-B0

Threshold:

- 0.91

Métricas apresentadas na aplicação:

| Métrica | Valor |
|---------|------:|
| ROC-AUC | 0.918 |
| Recall maligno | 76% |
| Acurácia | 91.8% |

---

# 13. Autores

- Rafael Reis Borges da Silva
- Victor Gabryel da Silva

Curso Superior de Tecnologia em Análise e Desenvolvimento de Sistemas

Instituto Federal de Educação, Ciência e Tecnologia de Pernambuco (IFPE) – Campus Jaboatão dos Guararapes

---

# 14. Observação

Este projeto possui finalidade acadêmica. O sistema serve como ferramenta de apoio à análise de imagens dermatológicas e não substitui o diagnóstico realizado por profissionais da saúde.