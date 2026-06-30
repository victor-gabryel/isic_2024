import streamlit as st

import torch
import pandas as pd
import os
import glob
import random

from PIL import Image

from torchvision import transforms


from src.config import DEVICE, BEST_MODEL
from src.model import create_model



# =====================================================
# CONFIG
# =====================================================


st.set_page_config(

    page_title="ISIC 2024 EfficientNet-B3",

    layout="wide"

)



THRESHOLD = 0.94


OUTPUT_DIR="outputs"


METADATA="data/ISIC_reduzido/metadata.csv"





# =====================================================
# MODELO
# =====================================================


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







# =====================================================
# TRANSFORM
# =====================================================


transform=transforms.Compose([


    transforms.Resize((224,224)),


    transforms.ToTensor(),


    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])








# =====================================================
# PREDIÇÃO
# =====================================================


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







# =====================================================
# DATASET
# =====================================================


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







# =====================================================
# BUSCAR IMAGEM
# =====================================================


def find_image(isic):


    extensoes=[

        "jpg",

        "jpeg",

        "png"

    ]


    for ext in extensoes:


        result=glob.glob(

            f"data/**/{isic}.{ext}",

            recursive=True

        )


        if result:

            return result[0]



    return None







# =====================================================
# TITULO
# =====================================================


st.title(
"🩺 ISIC 2024 - Classificação de Lesões Cutâneas"
)


st.markdown(

"""

Sistema desenvolvido utilizando Deep Learning com a arquitetura

**EfficientNet-B3**

para classificação automática de imagens dermatológicas.


O modelo realiza classificação binária:


🟢 Benigno


🔴 Maligno



Dataset utilizado:

**ISIC 2024 reduzido**

 contendo aproximadamente 10 mil imagens benignas e imagens malignas disponíveis.

"""

)







# =====================================================
# MÉTRICAS
# =====================================================


st.divider()


st.subheader(
"📊 Desempenho do Modelo"
)



a,b,c=st.columns(3)



a.metric(

"ROC-AUC",

"0.903"

)


b.metric(

"Recall maligno",

"73%"

)


c.metric(

"Acurácia",

"90%"

)









# =====================================================
# GRÁFICOS DO PROJETO (2 COLUNAS x 3 LINHAS)
# =====================================================

st.divider()

st.subheader(
    "📈 Gráficos do Projeto"
)


graficos = [

    (
        "Dataset Original ISIC 2024",
        "dataset_original.png"
    ),

    (
        "Dataset Reduzido ISIC 2024",
        "dataset_reduzido.png"
    ),

    (
        "Curva ROC-AUC",
        "roc_curve.png"
    ),

    (
        "Evolução da Acurácia",
        "accuracy.png"
    ),

    (
        "Evolução da Loss",
        "loss.png"
    ),

    (
        "Matriz de Confusão",
        "confusion_matrix.png"
    )

]



# cria grade 2 colunas

colunas = st.columns(2)



for index, (nome, arquivo) in enumerate(graficos):


    caminho = os.path.join(

        OUTPUT_DIR,

        arquivo

    )


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


            st.image(

                caminho,

                width=350

            )


        else:


            st.warning(

                f"Gráfico não encontrado: {arquivo}"

            )



    # depois de cada 2 gráficos cria nova linha

    if index % 2 == 1 and index != len(graficos)-1:


        colunas = st.columns(2)









# =====================================================
# IMAGEM INDIVIDUAL
# =====================================================


st.divider()


st.subheader(

"🔍 Testar imagem"

)



arquivo=st.file_uploader(

"Escolha uma imagem",

type=[

"jpg",

"jpeg",

"png"

]

)





if arquivo:


    img=Image.open(arquivo).convert("RGB")



    centro=st.columns([1,2,1])


    with centro[1]:


        st.image(

            img,

            width=220

        )


        pred,conf=predict(img)



        st.info(

f"""

IA:

**{pred}**


Confiança:

**{conf:.2%}**

"""

)









# =====================================================
# GALERIA TESTE
# =====================================================


st.divider()


st.subheader(

"🧪 Teste automático de imagens"

)





opcao=st.selectbox(

"Selecionar",

[

"10 imagens aleatórias",

"10 benignas",

"10 malignas",

"5 benignas + 5 malignas"

]

)





if st.button(

"🚀 GERAR TESTE NOVAMENTE"

):


    st.session_state["executar"]=True






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


                    st.image(

                    img,

                    width=120

                    )



                    st.caption(

f"""

ID:

{row.isic_id}


Real:

{row.classe}


IA:

{pred}


Conf:

{conf:.1%}


{resultado}

"""

)











# =====================================================
# SIDEBAR
# =====================================================


st.sidebar.title(

"Informações"

)


st.sidebar.write(

"""

Modelo:

EfficientNet-B3


Dataset:

ISIC 2024 reduzido


Classes:

Benigno / Maligno


Threshold:

0.94


"""

)