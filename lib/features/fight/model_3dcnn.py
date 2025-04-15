# lib/model_3dcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class My3DCNN(nn.Module):
    def __init__(self, num_classes=2, input_shape=(3, 16, 112, 112), dropout_prob=0.5): # Добавлен dropout_prob
        """
        :param num_classes: Число классов (2: noFight, fight)
        :param input_shape: (C, D, H, W) — ожидаемый размер входа (без батча)
        :param dropout_prob: Вероятность dropout перед полносвязными слоями
        """
        super(My3DCNN, self).__init__()
        self.input_shape = input_shape
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv3d(input_shape[0], 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # Не уменьшаем временную размерность сильно?

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        # Уменьшим временную размерность на последнем пулинге
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Проверим размер выхода после свёрток
        with torch.no_grad(): # Не считаем градиенты для dummy прогона
             dummy_input = torch.zeros(1, *self.input_shape)
             dummy_out = self._forward_features(dummy_input)
             flatten_dim = dummy_out.view(1, -1).size(1) # Надежный способ получить размер

        self.fc1 = nn.Linear(flatten_dim, 128)
        self.dropout = nn.Dropout(p=self.dropout_prob) # Слой Dropout
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_features(self, x):
        # Убедимся что вход правильной формы (на всякий случай)
        # Ожидаем [B, C, T, H, W]
        if x.shape[1:] != self.input_shape:
             # Попробуем изменить форму, если возможно (например, пропущен канал)
             # Или выдадим ошибку
             raise ValueError(f"Неправильная форма входа для _forward_features: {x.shape}, ожидалось B x {self.input_shape}")


        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Применяем Dropout
        x = self.fc2(x)
        return x