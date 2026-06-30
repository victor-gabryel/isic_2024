import torch
import torch.nn as nn

import timm

from src.config import MODEL_NAME



# ==========================================
# MODELO
# ==========================================

class SkinCancerModel(nn.Module):


    def __init__(self):

        super().__init__()



        # EfficientNet pré-treinada

        self.model = timm.create_model(

            MODEL_NAME,

            pretrained=True,

            num_classes=0

        )



        # Descobrir tamanho da saída

        features = self.model.num_features



        # Classificador final

        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(

                features,

                1

            )

        )



    def forward(self, x):


        x = self.model(x)


        x = self.classifier(x)


        return x.squeeze(1)





# ==========================================
# CRIAR MODELO
# ==========================================

def create_model(device):


    model = SkinCancerModel()


    model = model.to(device)


    return model