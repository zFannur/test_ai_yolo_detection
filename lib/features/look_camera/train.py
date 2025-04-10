
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
        data='fire_smoke.yaml',
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project='models',
        batch=9,
        name=f'yolo11_fire_smoke_detection_{MODEL_NAME.split(".")[0]}',
        pretrained=True,
        device=DEVICE,
        amp=False,  # отключено, т.к. CUDA и 4 ГБ
        warmup_epochs=1.0,  # быстрее старт
        lr0=0.005,  # чуть меньше стартовый LR
        weight_decay=0.0007,  # немного усилим регуляризацию
        mosaic=1.0,  # оставить включённым
        mixup=0.2,  # улучшит обобщение
        copy_paste=0.1,  # добавим синтетики
        erasing=0.4,  # оставить
        patience=10,  # модель остановится раньше, если нет прогресса
        verbose=True,  # подробный вывод
    )


if __name__ == '__main__':
    main()
