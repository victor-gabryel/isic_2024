# src/train.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import roc_auc_score

from config import *
from dataset import *
from model import create_model


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():

    set_seed(SEED)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    df = load_dataset()

    train_df, val_df, test_df = split_dataset(df)

    train_df = balance_train_set(train_df)

    train_dataset = SkinDataset(
        train_df,
        train_transform
    )

    val_dataset = SkinDataset(
        val_df,
        val_transform
    )

    classes = train_df["malignant"].astype(int).values

    class_counts = np.bincount(classes)

    weights = 1 / class_counts

    sample_weights = [
        weights[c]
        for c in classes
    ]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = create_model().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [POS_WEIGHT]
        ).to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    start_epoch = 0
    best_auc = 0

    train_losses = []
    val_aucs = []

    if os.path.exists(CHECKPOINT_PATH):

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        start_epoch = checkpoint["epoch"] + 1

        best_auc = checkpoint["best_auc"]

        train_losses = checkpoint["loss"]

        val_aucs = checkpoint["auc"]

        print(
            f"Continuando do epoch {start_epoch}"
        )

    stop_count = 0

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        model.train()

        total_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for images, labels in loop:

            images = images.to(DEVICE)

            labels = labels.to(
                DEVICE
            ).float().view(-1,1)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)

        train_losses.append(epoch_loss)

        model.eval()

        preds = []
        true = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)

                outputs = model(images)

                probs = torch.sigmoid(
                    outputs
                ).view(-1)

                preds.extend(
                    probs.cpu().numpy()
                )

                true.extend(
                    labels.numpy()
                )

        auc = roc_auc_score(
            true,
            preds
        )

        val_aucs.append(auc)

        scheduler.step(auc)

        print(
            f"Loss={epoch_loss:.4f} | AUC={auc:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": train_losses,
                "auc": val_aucs,
                "best_auc": best_auc
            },
            CHECKPOINT_PATH
        )

        if auc > best_auc:

            best_auc = auc

            stop_count = 0

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

        else:

            stop_count += 1

        if stop_count >= PATIENCE:

            print(
                "Early Stopping"
            )

            break

    plt.plot(train_losses)
    plt.title("Loss")
    plt.savefig(
        "outputs/loss_curve.png"
    )

    plt.close()

    plt.plot(val_aucs)
    plt.title("AUC")
    plt.savefig(
        "outputs/auc_curve.png"
    )

    plt.close()


if __name__ == "__main__":
    main()