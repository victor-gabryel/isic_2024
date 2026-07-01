import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch

from torchvision import transforms


# Caminho para o diretório de dados, incluindo imagens e arquivo CSV com metadados.
DATA_DIR = "data/ISIC_reduzido"

IMAGE_DIR = os.path.join(
    DATA_DIR,
    "images"
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "metadata.csv"
)


# Transformações aplicadas ao conjunto de treinamento, incluindo redimensionamento, aumento de dados (flip horizontal, rotação e ajuste de brilho/contraste), conversão para tensor e normalização com médias e desvios padrão específicos.
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


# Transformações aplicadas ao conjunto de validação, incluindo redimensionamento, conversão para tensor e normalização com médias e desvios padrão específicos.
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


# Dataset personalizado para o conjunto de dados ISIC. Ele herda da classe Dataset do PyTorch e implementa os métodos necessários para carregar imagens e rótulos a partir de um DataFrame.
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


    # Define o tamanho do dataset, retornando o número de amostras presentes no DataFrame.
    def __len__(self):

        return len(self.df)


    # Obtém um item do dataset com base no índice fornecido. Ele lê a imagem correspondente ao índice, aplica as transformações especificadas e retorna a imagem e o rótulo (maligno ou benigno) como tensores.
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


# Função para carregar o dataset a partir do arquivo CSV. Ela lê o CSV, adiciona uma coluna com os caminhos completos das imagens e retorna o DataFrame resultante.
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


# Função para criar DataLoaders para treinamento e validação. Ela divide o DataFrame em conjuntos de treinamento e validação, cria datasets personalizados usando a classe ISICDataset, e retorna DataLoaders para ambos os conjuntos.
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