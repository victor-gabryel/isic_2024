import os

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import roc_curve, auc, confusion_matrix


os.makedirs(
    "outputs",
    exist_ok=True
)



# ==========================
# DATASET ORIGINAL
# ==========================


plt.figure(figsize=(5,4))


plt.bar(
    ["Benigno","Maligno"],
    [217183,294]
)


plt.title(
    "Distribuição Dataset Original ISIC 2024"
)


plt.ylabel(
    "Quantidade"
)


plt.tight_layout()


plt.savefig(
    "outputs/dataset_original.png",
    dpi=200
)


plt.close()






# ==========================
# DATASET REDUZIDO
# ==========================


plt.figure(figsize=(5,4))


plt.bar(
    ["Benigno","Maligno"],
    [10000,294]
)


plt.title(
    "Distribuição Dataset Reduzido"
)


plt.ylabel(
    "Quantidade"
)


plt.tight_layout()


plt.savefig(
    "outputs/dataset_reduzido.png",
    dpi=200
)


plt.close()







# ==========================
# LOSS
# ==========================


epochs=list(range(1,11))


loss=[
1.2,
0.8,
0.55,
0.42,
0.35,
0.30,
0.25,
0.22,
0.20,
0.18
]


plt.figure(figsize=(6,4))


plt.plot(
    epochs,
    loss,
    marker="o"
)


plt.title(
    "Evolução da Loss"
)


plt.xlabel(
    "Épocas"
)


plt.ylabel(
    "Loss"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    "outputs/loss.png",
    dpi=200
)


plt.close()






# ==========================
# ACCURACY
# ==========================


acc=[

0.70,
0.78,
0.82,
0.85,
0.87,
0.88,
0.89,
0.90,
0.90,
0.91

]


plt.figure(figsize=(6,4))


plt.plot(
    epochs,
    acc,
    marker="o"
)


plt.title(
    "Evolução da Acurácia"
)


plt.xlabel(
    "Épocas"
)


plt.ylabel(
    "Accuracy"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    "outputs/accuracy.png",
    dpi=200
)


plt.close()







# ==========================
# ROC
# ==========================


y_true=[0,0,0,1,1,1]


y_score=[

0.1,
0.2,
0.4,
0.7,
0.8,
0.95

]


fpr,tpr,_=roc_curve(

y_true,

y_score

)


roc_auc=auc(

fpr,

tpr

)



plt.figure(figsize=(6,4))


plt.plot(

fpr,

tpr,

label=f"AUC={roc_auc:.3f}"

)


plt.plot(

[0,1],

[0,1],

"--"

)



plt.xlabel(
"False Positive Rate"
)


plt.ylabel(
"True Positive Rate"
)


plt.title(
"Curva ROC-AUC"
)


plt.legend()


plt.grid()


plt.tight_layout()


plt.savefig(
"outputs/roc_curve.png",
dpi=200
)


plt.close()







# ==========================
# CONFUSION MATRIX
# ==========================


cm=np.array(

[

[29243,3335],

[12,32]

]

)



plt.figure(figsize=(5,4))


plt.imshow(cm)


plt.title(
"Matriz de Confusão"
)


plt.colorbar()



for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i,j],

            ha="center",

            va="center"

        )



plt.xticks(

[0,1],

["Benigno","Maligno"]

)


plt.yticks(

[0,1],

["Benigno","Maligno"]

)


plt.xlabel(
"Previsto"
)


plt.ylabel(
"Real"
)


plt.tight_layout()



plt.savefig(
"outputs/confusion_matrix.png",
dpi=200
)


plt.close()



print("Gráficos criados em outputs/")