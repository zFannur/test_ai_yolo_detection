import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class My3DCNN(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.6):
        super(My3DCNN, self).__init__()
        # Загружаем предобученную модель
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)
        # Заменяем последний слой с учетом dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
