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



# ==============================
# CONFIG
# ==============================


THRESHOLD = 0.94


OUTPUT_DIR = "outputs/gradcam"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ==============================
# TRANSFORM
# ==============================


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





# ==============================
# ENCONTRAR CSV
# ==============================


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






# ==============================
# ENCONTRAR IMAGEM
# ==============================


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







# ==============================
# GRADCAM
# ==============================


class GradCAM:


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





    def forward_hook(
        self,
        module,
        input,
        output
    ):


        self.activations=output





    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):


        self.gradients=grad_output[0]







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







# ==============================
# SALVAR RESULTADO
# ==============================


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







# ==============================
# MAIN
# ==============================


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





    # CORREÇÃO IMPORTANTE
    # seu modelo é:
    # SkinCancerModel -> EfficientNet -> conv_head


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

ID:

{isic}


Real:

{"MALIGNO" if row.malignant==1 else "BENIGNO"}


Modelo:

{"MALIGNO" if pred==1 else "BENIGNO"}


Confiança:

{prob:.2%}


Threshold:

{THRESHOLD}


====================

""")





    print(

        "Grad-CAM finalizado!"

    )







if __name__=="__main__":


    main()