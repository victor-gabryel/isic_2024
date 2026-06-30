import kagglehub

path = kagglehub.dataset_download(
    "roscoekerby/isic-2024-permissive-training-input"
)

print(path)