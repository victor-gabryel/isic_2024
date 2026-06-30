import os
import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import *


# ==========================================
# TRANSFORMAÇÕES
# ==========================================

train_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.VerticalFlip(
        p=0.5
    ),

    A.RandomRotate90(
        p=0.5
    ),

    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),

    A.Normalize(),

    ToTensorV2()

])


val_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(),

    ToTensorV2()

])


# ==========================================
# CARREGAR DATASET
# ==========================================

def load_dataset():

    df = pd.read_csv(
        TRAIN_CSV
    )


    df["image_path"] = df["isic_id"].apply(

        lambda x:
        os.path.join(
            IMAGE_DIR,
            f"{x}.jpg"
        )

    )


    df = df[
        df["image_path"].apply(
            os.path.exists
        )
    ]


    # ==============================
    # BALANCEAMENTO DO DATASET
    # ==============================

    malignant = df[
        df["malignant"] == 1
    ]


    benign = df[
        df["malignant"] == 0
    ].sample(
        10000,
        random_state=42
    )


    df = pd.concat(
        [
            malignant,
            benign
        ]
    )


    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )


    print()

    print("Dataset balanceado:")

    print(
        df["malignant"].value_counts()
    )


    return df



# ==========================================
# DIVISÃO
# ==========================================

def split_dataset(df):


    train_df, val_df = train_test_split(

        df,

        test_size=0.2,

        stratify=df["malignant"],

        random_state=SEED

    )


    return (

        train_df.reset_index(drop=True),

        val_df.reset_index(drop=True)

    )



# ==========================================
# DATASET PYTORCH
# ==========================================

class SkinDataset(Dataset):


    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):


        row = self.df.iloc[index]


        image = Image.open(

            row["image_path"]

        ).convert(
            "RGB"
        )


        image = np.array(
            image
        )



        if self.transform:


            image = self.transform(

                image=image

            )["image"]



        label = torch.tensor(

            row["malignant"],

            dtype=torch.float32

        )



        return image, label



# ==========================================
# DATALOADERS
# ==========================================

def create_loaders(df):


    train_df, val_df = split_dataset(
        df
    )


    train_dataset = SkinDataset(

        train_df,

        train_transform

    )


    val_dataset = SkinDataset(

        val_df,

        val_transform

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=False

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=False

    )



    return train_loader, val_loader