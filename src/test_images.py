import torch
import os
import glob
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


from src.config import *
from src.model import create_model



# ==============================
# CONFIGURAÇÃO
# ==============================


NUM_IMAGES = 5


THRESHOLD = 0.94



OUTPUT_DIR = "outputs/test_visual"



os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)




# ==============================
# TRANSFORMAÇÃO
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

        "**/*.csv",

        recursive=True

    )


    print("CSV encontrados:")


    for f in files:

        print(f)



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



    raise Exception(
        "CSV com coluna malignant não encontrado"
    )







# ==============================
# CARREGAR AMOSTRAS
# ==============================


def load_samples():


    csv=get_csv()


    print(
        "Dataset:",
        csv
    )



    df=pd.read_csv(csv)



    malignant=df[

        df["malignant"]==1

    ].sample(

        NUM_IMAGES

    )



    benign=df[

        df["malignant"]==0

    ].sample(

        NUM_IMAGES

    )



    samples=pd.concat(

        [

            malignant,

            benign

        ]

    )



    return samples.sample(

        frac=1

    )







# ==============================
# ENCONTRAR IMAGEM
# ==============================


def find_image(isic_id):


    patterns=[


        f"data/**/{isic_id}.jpg",


        f"data/**/{isic_id}.jpeg",


        f"data/**/{isic_id}.png"


    ]



    for pattern in patterns:


        result=glob.glob(

            pattern,

            recursive=True

        )


        if len(result)>0:


            return result[0]



    return None







# ==============================
# PREDIÇÃO
# ==============================


def predict(model,path):


    image=Image.open(

        path

    ).convert(

        "RGB"

    )



    tensor=transform(

        image

    )



    tensor=tensor.unsqueeze(

        0

    ).to(

        DEVICE

    )




    with torch.no_grad():


        output=model(

            tensor

        )


        probability=torch.sigmoid(

            output

        ).item()




    prediction=int(

        probability >= THRESHOLD

    )



    return image,prediction,probability







# ==============================
# MAIN
# ==============================


def main():


    print("================")
    print("Teste visual")
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




    samples=load_samples()



    correct=0


    total=0





    for _,row in samples.iterrows():



        image_path=find_image(

            row["isic_id"]

        )



        if image_path is None:


            print(

                "Imagem não encontrada:",

                row["isic_id"]

            )


            continue




        image,pred,confidence=predict(

            model,

            image_path

        )



        real=int(

            row["malignant"]

        )



        total+=1




        if pred==real:


            correct+=1




        real_text=(

            "MALIGNO"

            if real

            else

            "BENIGNO"

        )




        pred_text=(

            "MALIGNO"

            if pred

            else

            "BENIGNO"

        )





        result=(

            "ACERTO"

            if pred==real

            else

            "ERRO"

        )





        plt.figure(

            figsize=(4,4)

        )



        plt.imshow(

            image

        )



        plt.axis(

            "off"

        )



        plt.title(

f"""
Real: {real_text}

Modelo: {pred_text}

Confiança: {confidence:.2%}

{result}

Threshold: {THRESHOLD}
"""

        )




        save_path=os.path.join(

            OUTPUT_DIR,

            f"{row['isic_id']}.png"

        )



        plt.savefig(

            save_path,

            dpi=150,

            bbox_inches="tight"

        )



        plt.close()



        print(

f"""
{row['isic_id']}

Real:
{real_text}

Modelo:
{pred_text}

Confiança:
{confidence:.2%}

{result}

"""

        )





    print("====================")

    print("Resultado final")


    print(

f"""

Acertos:
{correct}/{total}


Precisão:
{correct/total:.2%}


Threshold usado:
{THRESHOLD}

"""

    )



    print("====================")






if __name__=="__main__":


    main()