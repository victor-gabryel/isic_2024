import os
import shutil
import pandas as pd


INPUT_IMAGES = "data/ISIC_2024_Permissive_Training_Input"

CSV = "data/ISIC_2024_Permissive_Training_GroundTruth.csv"


OUTPUT = "data/ISIC_reduzido/images"


os.makedirs(
    OUTPUT,
    exist_ok=True
)


df = pd.read_csv(CSV)


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
)


print(df["malignant"].value_counts())


for image_id in df["isic_id"]:

    src = os.path.join(
        INPUT_IMAGES,
        f"{image_id}.jpg"
    )


    dst = os.path.join(
        OUTPUT,
        f"{image_id}.jpg"
    )


    if os.path.exists(src):

        shutil.copy(
            src,
            dst
        )


df.to_csv(
    "data/ISIC_reduzido/metadata.csv",
    index=False
)


print("Dataset reduzido criado!")