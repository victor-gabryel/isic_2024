import torch.nn as nn
import timm



class SkinCancerModel(nn.Module):


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



    def forward(self,x):


        x = self.model(x)

        x = self.classifier(x)


        return x.squeeze(1)





def create_model(device):


    model = SkinCancerModel()


    return model.to(device)