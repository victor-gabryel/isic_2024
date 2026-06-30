import pandas as pd
import os

csv = "data/ISIC_reduzido/metadata.csv"
img = "data/ISIC_reduzido/images"

df = pd.read_csv(csv)

print(df.head())

print("\nClasses:")
print(df["malignant"].value_counts())

print("\nImagens:")

files = os.listdir(img)

print("Total:",len(files))

print(files[:5])