import torch
import numpy as np

from sklearn.metrics import f1_score

from src.config import *
from src.dataset import load_dataset, create_loaders
from src.model import create_model



def main():


    print("====================")
    print("Encontrando Threshold")
    print("====================")


    df = load_dataset()


    _, val_loader = create_loaders(df)



    model = create_model(
        DEVICE
    )


    checkpoint=torch.load(
        BEST_MODEL,
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    model.to(DEVICE)

    model.eval()



    probs=[]

    labels=[]



    with torch.no_grad():


        for images,y in val_loader:


            images=images.to(DEVICE)


            output=model(images)


            p=torch.sigmoid(output)



            probs.extend(
                p.cpu().numpy()
            )


            labels.extend(
                y.numpy()
            )



    probs=np.array(probs)

    labels=np.array(labels)



    best_threshold=0

    best_f1=0



    for t in np.arange(
        0.05,
        0.95,
        0.01
    ):


        pred=(probs>=t).astype(int)


        f1=f1_score(
            labels,
            pred
        )


        if f1>best_f1:

            best_f1=f1

            best_threshold=t



    print()

    print(
        "Melhor threshold:",
        best_threshold
    )


    print(
        "Melhor F1:",
        best_f1
    )



    with open(
        "outputs/best_threshold.txt",
        "w"
    ) as f:


        f.write(
            str(best_threshold)
        )



if __name__=="__main__":

    main()