# python features/open_product/train.py

import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
from lib.core.config import MODEL_NAME, EPOCHS, IMG_SIZE, DEVICE


def main():
    # Загрузка предобученной модели YOLO11
    model = YOLO(MODEL_NAME)

    # Перенос модели на устройство
    model.to(DEVICE)

    print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

    # Обучение модели на датасете
    model.train(
        data='open_product.yaml',
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project='models',
        batch=9,  # Уменьшите до значения, которое подходит вашей системе (если оперативки не хватает)
        name=f'yolo11_open_product_detection_{MODEL_NAME.split(".")[0]}',
        pretrained=True,
        device=DEVICE,
        amp=False,  # Используем смешанную точность для оптимизации(если cpu при cudo должно быть False)
    )


if __name__ == '__main__':
    main()
