import torch

# Define o dispositivo a ser usado para treinamento e inferência. Se uma GPU estiver disponível, ela será usada; caso contrário, o código será executado na CPU.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Configurações do modelo e treinamento, incluindo o nome do modelo, tamanho da imagem, tamanho do lote, número de épocas, taxa de aprendizado e caminho para o melhor modelo salvo.
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
BEST_MODEL = "checkpoints/best_model.pth"