# src/model.py

import timm
import torch.nn as nn


def create_model():

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=True
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(
            in_features,
            1
        )
    )

    return model