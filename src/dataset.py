import os

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import torch

from torchvision import transforms



# ===============================
# CONFIG
# ===============================

DATA_DIR = "data/ISIC_reduzido"

IMAGE_DIR = os.path.join(
    DATA_DIR,
    "images"
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "metadata.csv"
)



# ===============================
# TRANSFORMS
# ===============================


train_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])



val_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])



# ===============================
# DATASET
# ===============================


class ISICDataset(Dataset):


    def __init__(

        self,
        dataframe,
        transform=None

    ):


        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self,index):


        row = self.df.iloc[index]


        img_path = row["image_path"]


        image = Image.open(
            img_path
        ).convert(
            "RGB"
        )


        label = torch.tensor(

            row["malignant"],

            dtype=torch.float32

        )


        if self.transform:

            image = self.transform(image)



        return image,label





# ===============================
# LOAD DATASET
# ===============================


def load_dataset():


    df = pd.read_csv(
        CSV_PATH
    )


    df["image_path"] = (

        IMAGE_DIR
        + "/"
        + df["isic_id"]
        + ".jpg"

    )



    return df





# ===============================
# LOADERS
# ===============================


def create_loaders(

    df,

    batch_size=32

):


    train_df, val_df = train_test_split(

        df,

        test_size=0.2,

        stratify=df["malignant"],

        random_state=42

    )



    train_dataset = ISICDataset(

        train_df,

        train_transform

    )


    val_dataset = ISICDataset(

        val_df,

        val_transform

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0

    )


    return train_loader,val_loader