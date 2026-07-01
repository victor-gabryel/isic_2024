import torch
import torch.nn as nn

from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import json
import os

from src.config import *
from src.dataset import load_dataset, create_loaders
from src.model import create_model


# Configurações
os.makedirs(
    "checkpoints",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)

# Quantas épocas sem melhorar antes de parar
EARLY_STOPPING_PATIENCE = 5


# Treina o modelo por uma época e retorna a loss e AUC do treino
def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    model.train()
    losses=[]
    preds=[]
    labels_all=[]

    for images, labels in tqdm(loader):

        images=images.to(device)
        labels=labels.float().to(device)
        optimizer.zero_grad()
        output=model(images)

        loss=criterion(
            output,
            labels
        )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        prob=torch.sigmoid(output)
        
        preds.extend(
            prob.detach()
            .cpu()
            .numpy()
        )

        labels_all.extend(
            labels.cpu()
            .numpy()
        )

    auc=roc_auc_score(
        labels_all,
        preds
    )

    return sum(losses)/len(losses), auc


# Valida o modelo e retorna a loss e AUC da validação
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()
    losses=[]
    preds=[]
    labels_all=[]

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images=images.to(device)
            labels=labels.float().to(device)
            output=model(images)

            loss=criterion(
                output,
                labels
            )

            losses.append(loss.item())
            prob=torch.sigmoid(output)

            preds.extend(
                prob.cpu()
                .numpy()
            )

            labels_all.extend(
                labels.cpu()
                .numpy()
            )
            
    auc=roc_auc_score(
        labels_all,
        preds
    )
    
    return sum(losses)/len(losses), auc






# Função principal
def main():
    print("===================")
    print("Treino ISIC 2024")
    print("===================")

    # carregar dataset
    df=load_dataset()

    train_loader,val_loader=create_loaders(
        df
    )

    # modelo EfficientNet-B0
    model=create_model(
        DEVICE
    )

    # peso da classe maligna
    malignant=len(
        df[df.malignant==1]
    )

    benign=len(
        df[df.malignant==0]
    )

    pos_weight=torch.tensor(
        benign/malignant
    ).to(DEVICE)

    print(
        "Pos weight:",
        pos_weight
    )

    # função de perda
    criterion=nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer=AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )
    
    best_auc=0
    patience=0

    history={
        "epoch":[],
        "train_loss":[],
        "val_loss":[],
        "train_auc":[],
        "val_auc":[]
    }

    for epoch in range(EPOCHS):
        
        print(f"\nÉpoca {epoch+1}/{EPOCHS}")

        train_loss,train_auc=train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        val_loss,val_auc=validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        print(
            f"""
            Treino
            Loss: {train_loss:.4f}
            AUC: {train_auc:.4f}
            
            Validação
            Loss: {val_loss:.4f}
            AUC: {val_auc:.4f}
            """
        )

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        # salvar melhor modelo
        if val_auc > best_auc:
            best_auc=val_auc
            patience=0

            torch.save(
                {
                "model": model.state_dict(),
                "auc": best_auc,
                "epoch": epoch+1
                },
                BEST_MODEL
            )

            print("Melhor modelo salvo!")
        else:
            patience+=1
            print(
                f"Sem melhora {patience}/{EARLY_STOPPING_PATIENCE}"
            )

        # parar se não melhorar
        if patience >= EARLY_STOPPING_PATIENCE:
            print("Early stopping ativado")
            break

    # salvar histórico
    with open(
        "outputs/history.json",
        "w"
    ) as f:
        json.dump(
            history,
            f,
            indent=4
        )

    print("===================")
    print("Treino finalizado")
    print(f"Melhor AUC: {best_auc:.4f}")
    print("===================")

if __name__=="__main__":
    main()