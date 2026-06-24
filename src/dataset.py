# src/dataset.py

import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    DATASET_CSV,
    IMAGES_PATH,
    IMAGE_SIZE,
    SEED
)

# =====================================
# CARREGAR DATASET
# =====================================

def load_dataset():

    df = pd.read_csv(DATASET_CSV)

    df = df.dropna(subset=["malignant"])

    df["image_path"] = df["isic_id"].apply(
        lambda x: os.path.join(
            IMAGES_PATH,
            f"{x}.jpg"
        )
    )

    df = df[
        [
            "image_path",
            "malignant"
        ]
    ]

    return df


# =====================================
# SPLIT
# =====================================

def split_dataset(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["malignant"],
        random_state=SEED
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.176,
        stratify=train_df["malignant"],
        random_state=SEED
    )

    return train_df, val_df, test_df


# =====================================
# BALANCEAMENTO
# =====================================

def balance_train_set(train_df):

    train_benign = train_df[
        train_df["malignant"] == 0
    ]

    train_malignant = train_df[
        train_df["malignant"] == 1
    ]

    quantidade_benignas = (
        len(train_malignant) * 3
    )

    train_benign = train_benign.sample(
        n=quantidade_benignas,
        random_state=SEED
    )

    train_df = pd.concat(
        [
            train_benign,
            train_malignant
        ]
    )

    train_df = train_df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    return train_df


# =====================================
# TRANSFORMAÇÕES
# =====================================

train_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.Rotate(
        limit=30,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        p=0.5
    ),

    A.CLAHE(
        p=0.3
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()

])

val_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()

])


# =====================================
# DATASET
# =====================================

class SkinDataset(Dataset):

    def __init__(
        self,
        df,
        transform=None
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        if not os.path.exists(
            row["image_path"]
        ):

            raise FileNotFoundError(
                row["image_path"]
            )

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = np.array(image)

        label = row["malignant"]

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        return (
            image,
            torch.tensor(label).float()
        )