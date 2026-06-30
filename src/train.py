import torch
import torch.nn as nn

from torch.optim import AdamW

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from src.config import *

from src.dataset import load_dataset, create_loaders

from src.model import create_model

# ==========================================
# TREINAMENTO
# ==========================================


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    predictions = []

    targets = []

    for images, labels in tqdm(loader):

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(outputs)

        predictions.extend(probs.detach().cpu().numpy())

        targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(targets, predictions)

    return (total_loss / len(loader), auc)


# ==========================================
# VALIDAÇÃO
# ==========================================


def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    predictions = []

    targets = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            predictions.extend(probs.cpu().numpy())

            targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(targets, predictions)

    return (total_loss / len(loader), auc)


# ==========================================
# MAIN
# ==========================================


def main():
    
    print("==============================")
    print("Configuração ISIC 2024")
    print("==============================")
    print(
        "Dispositivo:",
        DEVICE
    )
    print(
        "Modelo:",
        MODEL_NAME
    )
    print(
        "Imagens:",
        IMAGE_DIR
    )
    print("==============================")
    

    print("Carregando dataset...")

    df = load_dataset()

    train_loader, val_loader = create_loaders(df)

    print("Criando modelo...")

    model = create_model(DEVICE)

    pos_weight = torch.tensor(
        [
            len(df[df["malignant"] == 0]) /
            len(df[df["malignant"] == 1])
        ]
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    best_auc = 0

    patience = 3

    counter = 0


    for epoch in range(EPOCHS):

        print()

        print(f"Época {epoch+1}/{EPOCHS}")

        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(f"""
                    Treino:
                    Loss: {train_loss:.4f}
                    AUC: {train_auc:.4f}

                    Validação:
                    Loss: {val_loss:.4f}
                    AUC: {val_auc:.4f}
                    """)

        if val_auc > best_auc:

            best_auc = val_auc

            counter = 0

            torch.save(model.state_dict(), BEST_MODEL)

            print("Modelo salvo!")

        else:

            counter += 1

            print(f"Sem melhora: {counter}/{patience}")

            if counter >= patience:

                print("Early stopping!")

                break


if __name__ == "__main__":

    main()
