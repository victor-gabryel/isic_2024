import torch
import os
import glob
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from src.config import *
from src.model import create_model

# Configurações do Grad-CAM
THRESHOLD = 0.91
OUTPUT_DIR = "outputs/gradcam"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# O transform é usado para pré-processar as imagens antes de passá-las pelo modelo. Ele redimensiona a imagem para 224x224 pixels, converte para tensor e normaliza com os valores médios e desvios padrão do ImageNet.
transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
    
])


# Encontra o arquivo CSV do dataset ISIC 2024. Ele procura recursivamente na pasta "data" por arquivos CSV e verifica se a coluna "malignant" está presente. Se encontrar, retorna o caminho do arquivo; caso contrário, lança um erro.

def get_csv():

    files = glob.glob(
        "data/**/*.csv",
        recursive=True
    )

    for file in files:

        try:

            df=pd.read_csv(
                file,
                nrows=5
            )

            if "malignant" in df.columns:
                return file

        except:

            pass

    raise FileNotFoundError(
        "CSV ISIC não encontrado"
    )


# Encontra a imagem correspondente a um determinado ID ISIC. Ele procura por arquivos com extensões jpg, jpeg e png na pasta "data" e suas subpastas. Se encontrar, retorna o caminho do arquivo; caso contrário, retorna None.
def find_image(isic_id):

    extensions=[
        "jpg",
        "jpeg",
        "png"
    ]

    for ext in extensions:

        result=glob.glob(
            f"data/**/{isic_id}.{ext}",
            recursive=True
        )

        if result:
            return result[0]

    return None


# O Grad-CAM (Gradient-weighted Class Activation Mapping) é uma técnica de visualização que destaca as regiões da imagem que são mais importantes para a decisão do modelo. Ele utiliza os gradientes das classes em relação às ativações da última camada convolucional para gerar um mapa de calor que indica quais partes da imagem contribuíram mais para a previsão do modelo.
class GradCAM:

    # O GradCAM é inicializado com o modelo e a camada alvo (geralmente a última camada convolucional). Ele registra hooks para capturar as ativações e gradientes durante a passagem para frente e para trás do modelo. O método generate calcula o mapa de calor usando os gradientes e ativações, normaliza o mapa e o redimensiona para o tamanho da imagem original.
    def __init__(self,model,layer):

        self.model=model
        self.activations=None
        self.gradients=None

        layer.register_forward_hook(
            self.forward_hook
        )

        layer.register_full_backward_hook(
            self.backward_hook
        )

    # O método forward_hook é chamado durante a passagem para frente do modelo e armazena as ativações da camada alvo. O método backward_hook é chamado durante a passagem para trás e armazena os gradientes em relação às ativações da camada alvo. Esses hooks permitem que o Grad-CAM capture as informações necessárias para gerar o mapa de calor.
    def forward_hook(
        self,
        module,
        input,
        output
    ):

        self.activations=output

    # O método backward_hook é chamado durante a passagem para trás do modelo e armazena os gradientes em relação às ativações da camada alvo. Esses gradientes são usados para calcular os pesos que indicam a importância de cada canal de ativação na decisão do modelo.
    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients=grad_output[0]

    # O método generate calcula o mapa de calor do Grad-CAM. Ele realiza a passagem para trás do modelo para obter os gradientes, calcula os pesos médios dos gradientes, combina esses pesos com as ativações da camada alvo e aplica a função ReLU para obter o mapa de calor final. O mapa é então normalizado e redimensionado para o tamanho da imagem original.
    def generate(self,output):
        
        self.model.zero_grad()
        output.backward()
        gradients=self.gradients[0]
        activations=self.activations[0]

        weights=torch.mean(
            gradients,
            dim=(1,2)
        )

        cam=torch.zeros(
            activations.shape[1:],
            device=DEVICE
        )


        for i,w in enumerate(weights):
            cam += w * activations[i]

        cam=torch.relu(cam)
        cam=cam.detach().cpu().numpy()

        cam=cv2.resize(
            cam,
            (224,224)
        )

        cam=(
            cam-cam.min()
        )/(
            cam.max()-cam.min()+1e-8
        )
        return cam


# O método save_gradcam salva a imagem original e o mapa de calor do Grad-CAM sobreposto em um arquivo PNG. Ele aplica um mapa de cores ao mapa de calor, combina com a imagem original e salva o resultado em uma pasta de saída especificada.
def save_gradcam(
    image,
    cam,
    name
):

    img=np.array(image)

    heatmap=cv2.applyColorMap(
        np.uint8(255*cam),
        cv2.COLORMAP_JET
    )

    heatmap=cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay=(
        0.5*img+
        0.5*heatmap
    )

    overlay=np.uint8(
        overlay
    )

    plt.figure(
        figsize=(8,4)
    )

    plt.subplot(
        1,
        2,
        1
    )


    plt.imshow(img)

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(overlay)

    plt.title(
        "Grad-CAM"
    )

    plt.axis(
        "off"
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{name}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# O método main é o ponto de entrada do script. Ele cria o modelo, carrega os pesos do melhor modelo salvo, define a camada alvo para o Grad-CAM, gera mapas de calor para amostras selecionadas do dataset e salva as imagens resultantes. Ele também imprime informações sobre cada amostra, incluindo o ID, a classe real, a previsão do modelo e a confiança da previsão.
def main():

    print("================")
    print("Grad-CAM ISIC 2024")
    print("================")

    model=create_model(
        DEVICE
    )

    checkpoint=torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(
        DEVICE
    )

    model.eval()

    target_layer = model.model.conv_head

    gradcam=GradCAM(
        model,
        target_layer
    )

    csv=get_csv()

    print(
        "Dataset:",
        csv
    )

    df=pd.read_csv(csv)

    samples=pd.concat([

        df[df.malignant==1].sample(
            5,
            random_state=42
        ),

        df[df.malignant==0].sample(
            5,
            random_state=42
        )

    ])


    for _,row in samples.iterrows():

        isic=row["isic_id"]

        path=find_image(isic)

        if path is None:

            print(
                "Imagem não encontrada:",
                isic
            )

            continue


        image=Image.open(
            path
        ).convert(
            "RGB"
        )

        x=transform(
            image
        ).unsqueeze(0).to(
            DEVICE
        )

        output=model(x)

        prob=torch.sigmoid(
            output
        ).item()

        pred=int(
            prob >= THRESHOLD
        )

        cam=gradcam.generate(
            output
        )

        save_gradcam(
            image.resize(
                (224,224)
            ),
            cam,
            isic
        )

        print(f"""
                ====================
                ID: {isic}
                Real: {"MALIGNO" if row.malignant==1 else "BENIGNO"}
                Modelo: {"MALIGNO" if pred==1 else "BENIGNO"}
                Confiança: {prob:.2%}
                Threshold: {THRESHOLD}
                ====================
                """)

    print(
        "Grad-CAM finalizado!"
    )

if __name__=="__main__":
    main()