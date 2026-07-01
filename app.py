import streamlit as st
import torch
import pandas as pd
import os
import glob

from PIL import Image
from torchvision import transforms
from src.config import DEVICE, BEST_MODEL
from src.model import create_model


# Configurações do Streamlit

st.set_page_config(
    page_title="ISIC 2024 EfficientNet-B0",
    layout="wide"
)

THRESHOLD = 0.91
OUTPUT_DIR="outputs"
METADATA="data/ISIC_reduzido/metadata.csv"


# Modelo EfficientNet-B0 pré-treinado no conjunto de dados ISIC 2024. O modelo é carregado a partir de um checkpoint salvo em BEST_MODEL. Ele é configurado para avaliação (modo eval) e movido para o dispositivo especificado (CPU ou GPU).


# Carrega o modelo EfficientNet-B0
@st.cache_resource
def load_model():
    model=create_model(DEVICE)
    checkpoint=torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()
    return model

model=load_model()


# Transformações aplicadas às imagens de entrada antes da predição. As imagens são redimensionadas para 224x224 pixels, convertidas em tensores e normalizadas com médias e desvios padrão específicos.
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# Prediz a classe de uma imagem usando o modelo carregado. A imagem é transformada, passada pelo modelo e a probabilidade de ser maligno é calculada. Com base em um limiar (THRESHOLD), a função retorna a classe prevista ("MALIGNO" ou "BENIGNO") e a probabilidade associada.
def predict(img):
    x=transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output=model(x)
        prob=torch.sigmoid(output).item()
    classe = (
        "MALIGNO"
        if prob >= THRESHOLD
        else
        "BENIGNO"
    )

    return classe,prob


# Dataframe do dataset é carregado a partir de um arquivo CSV especificado em METADATA. A função adiciona uma coluna "classe" que classifica as amostras como "MALIGNO" ou "BENIGNO" com base na coluna "malignant". O dataframe resultante é retornado para uso posterior na aplicação.
@st.cache_data
def load_dataset():

    df=pd.read_csv(METADATA)
    
    df["classe"]=df["malignant"].apply(
        lambda x:
        "MALIGNO"
        if x==1
        else
        "BENIGNO"
    )

    return df

df=load_dataset()


# Busca o caminho de uma imagem no dataset com base no ID ISIC fornecido. A função procura por arquivos com extensões comuns (jpg, jpeg, png) na pasta "data" e retorna o caminho completo da primeira correspondência encontrada. Se nenhuma imagem for encontrada, a função retorna None.
def find_image(isic):

    extensoes=["jpg", "jpeg", "png"]

    for ext in extensoes:
        result=glob.glob(
            f"data/**/{isic}.{ext}",
            recursive=True
        )

        if result:
            return result[0]

    return None


# Titulo principal da aplicação Streamlit, que indica que se trata de um projeto de classificação de lesões cutâneas utilizando o conjunto de dados ISIC 2024 e a arquitetura EfficientNet-B0.

st.title(
    "ISIC 2024 - Classificação de Lesões Cutâneas"
    )


st.markdown(

"""

Aplicação de Deep Learning para classificação automática de lesões cutâneas utilizando a arquitetura EfficientNet-B0 e o conjunto de dados ISIC 2024 reduzido.

O modelo realiza classificação binária:

Classes:

- **Benigno**: Lesões cutâneas não cancerígenas.
- **Maligno**: Lesões cutâneas cancerígenas.

Dataset utilizado:

**ISIC 2024 reduzido**

 contendo aproximadamente 10 mil imagens benignas e imagens malignas disponíveis.

"""
)


# Metricas de desempenho do modelo, incluindo ROC-AUC, Recall para a classe maligno e Acurácia geral. Essas métricas são exibidas em três colunas na interface do Streamlit, permitindo uma visualização rápida do desempenho do modelo.
st.divider()

st.subheader("Desempenho do Modelo")

a,b,c=st.columns(3)

a.metric("ROC-AUC","0.918")
b.metric("Recall maligno","76%")
c.metric("Acurácia","91.8%")


# Graficos do projeto, incluindo distribuição das classes no dataset original e reduzido, curva ROC-AUC, evolução da acurácia e loss durante o treinamento, e matriz de confusão. Cada gráfico é exibido em uma grade de 2 colunas por 3 linhas na interface do Streamlit, permitindo uma análise visual completa do desempenho do modelo e da distribuição das classes.
st.divider()

st.subheader("📈 Gráficos do Projeto")

graficos = [
    ("Dataset Original ISIC 2024","dataset_original.png"),
    ("Dataset Reduzido ISIC 2024","dataset_reduzido.png"),
    ("Curva ROC-AUC","roc_curve.png"),
    ("Evolução da Loss","loss.png"),
    ("Matriz de Confusão","confusion_matrix.png")
]


# cria grade 2 colunas
colunas = st.columns(2)

for index, (nome, arquivo) in enumerate(graficos):

    caminho = os.path.join(OUTPUT_DIR,arquivo)
    coluna = colunas[index % 2]

    with coluna:
        if os.path.exists(caminho):
            st.markdown(
                f"""
                <h4 style="
                text-align:center;
                margin-top:20px;
                ">
                {nome}
                </h4>
                """,
                unsafe_allow_html=True
            )

            st.image(caminho, use_container_width=True)
        else:
            st.warning(
                f"Gráfico não encontrado: {arquivo}"
            )

    # depois de cada 2 gráficos cria nova linha
    if index % 2 == 1 and index != len(graficos)-1:
        colunas = st.columns(2)


# Imagens de exemplo do dataset, incluindo uma imagem benigna e uma imagem maligna. As imagens são exibidas lado a lado na interface do Streamlit, permitindo uma visualização rápida das diferenças entre as classes.

st.divider()

st.subheader("Testar imagem")

arquivo=st.file_uploader(
    "Escolha uma imagem",
    type=["jpg","jpeg","png"]
)


if arquivo:
    img=Image.open(arquivo).convert("RGB")
    centro=st.columns([1,2,1])

    with centro[1]:
        st.image(img,width=220)
        pred,conf=predict(img)
        st.info(
                f"""
                IA:
                **{pred}**
                Confiança:
                **{conf:.2%}**
                """
                )


# Galeria de imagens de teste, que permite ao usuário selecionar diferentes conjuntos de imagens para avaliação automática pelo modelo. As opções incluem 10 imagens aleatórias, 10 imagens benignas, 10 imagens malignas ou uma combinação de 5 benignas e 5 malignas. O usuário pode gerar um novo conjunto de teste clicando em um botão, e as imagens selecionadas são exibidas com suas respectivas predições e níveis de confiança.
st.divider()

st.subheader("Teste automático de imagens")

opcao=st.selectbox(
    "Selecionar",
    [
        "10 imagens aleatórias",
        "10 benignas",
        "10 malignas",
        "5 benignas + 5 malignas"
    ]
)

if st.button("GERAR TESTE NOVAMENTE"):
    st.session_state["executar"]=True


# Escolhe um conjunto de imagens com base na opção selecionada pelo usuário. Dependendo da escolha, a função retorna um subconjunto do dataframe contendo 10 imagens aleatórias, 10 imagens benignas, 10 imagens malignas ou uma combinação de 5 benignas e 5 malignas. O conjunto de imagens escolhido é então utilizado para avaliação automática pelo modelo.
def escolher():

    if opcao=="10 imagens aleatórias":
        return df.sample(10)

    if opcao=="10 benignas":
        return df[df.malignant==0].sample(10)

    if opcao=="10 malignas":
        return df[df.malignant==1].sample(10)

    return pd.concat([
        df[df.malignant==0].sample(5),
        df[df.malignant==1].sample(5)
    ])


if st.session_state.get("executar"):
    imagens=escolher()
    imagens=imagens.reset_index(drop=True)

    linhas=[
        imagens.iloc[i:i+5]
        for i in range(
            0,
            len(imagens),
            5
        )
    ]

    for linha in linhas:
        cols=st.columns(5)
        for col,(_,row) in zip(cols,linha.iterrows()):
            
            path=find_image(
                row.isic_id
            )

            if path:
                img=Image.open(path).convert("RGB")

                pred,conf=predict(img)

                resultado=(
                    "ACERTO ✅"
                    if pred==row.classe
                    else
                    "ERRO ❌"
                )

                with col:
                    st.image(img,width=120)

                    st.markdown(f"""
                        **ID:** {row.isic_id}
                        **Real:** {row.classe}
                        **IA:** {pred}
                        **Confiança:** {conf:.1%}
                        **{resultado}**
                        """)


# Sidebar com informações sobre o modelo, dataset, classes e threshold utilizado na classificação. A barra lateral fornece uma visão geral rápida do projeto, incluindo detalhes sobre a arquitetura do modelo (EfficientNet-B0), o conjunto de dados utilizado (ISIC 2024 reduzido), as classes de classificação (Benigno e Maligno) e o limiar de decisão (0.91) aplicado para determinar a classe prevista.

st.sidebar.title("Informações")

st.sidebar.write(
    """
    Modelo:

    EfficientNet-B0

    Dataset:

    ISIC 2024 reduzido

    Classes:

    Benigno / Maligno

    Threshold:

    0.91

    """
)