import torch.nn as nn
import timm
class SkinCancerModel(nn.Module):

    # Inicializa o modelo para detecção de câncer de pele
    def __init__(self):

        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        features = self.model.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(
                features,
                1
            )
        )

    # Define o método forward para a passagem de dados pelo modelo. Ele recebe uma entrada x, passa pelo modelo base e pelo classificador, e retorna a saída final.
    def forward(self,x):
        x = self.model(x)
        x = self.classifier(x)
        return x.squeeze(1)

# Cria uma instância do modelo SkinCancerModel e a move para o dispositivo especificado (CPU ou GPU). A função create_model é usada para encapsular a criação do modelo e garantir que ele esteja no dispositivo correto.
def create_model(device):
    model = SkinCancerModel()
    return model.to(device)