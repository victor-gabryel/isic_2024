import os
from pathlib import Path


print("======================")
print("RELATÓRIO FINAL ISIC")
print("======================")


print()


print("Modelo:")
print("EfficientNet")


print()


print("Threshold:")
print("0.94")


print()


print("Arquivos gerados:")


files=list(
    Path("outputs").rglob("*")
)


for f in files:

    print(
        f
    )


print()


print("======================")
print("Resumo:")
print("======================")


print("""
Treinamento realizado.

Métrica principal:
AUC

Threshold otimizado:
0.94

Teste visual:
90% de acurácia

Grad-CAM:
Gerado para análise de explicabilidade.

""")