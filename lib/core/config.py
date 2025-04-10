# config.py
import torch

# обученные модели определения лица и сигареты
# 'yolo11n.pt' 20 эпох
# 'yolo11s.pt' 10 эпох
# 'yolo11m.pt' 20 эпох

# обученные модели определения падения
# 'yolo11s.pt' 20 эпох

MODEL_NAME = 'yolo11s.pt'  # Название модели
EPOCHS = 30  # Количество эпох
IMG_SIZE = 640  # Размер изображений
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Автоматический выбор устройства
SMOKE_DETECTION_DISTANCE = 100  # Дистанция обнаружения сигареты
DETECTION_DISTANCE = 100
