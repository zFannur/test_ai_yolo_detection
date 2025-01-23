from ultralytics import YOLO
import cv2
import os
import numpy as np
from config import MODEL_NAME, DEVICE, DETECTION_DISTANCE

# Путь к обученной модели
MODEL_PATH = f'models/yolo11_count_product_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'

# Загрузка обученной модели
model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

# Инициализация видеопотока из файла
VIDEO_PATH = 'datasets/count_product/video/train.mp4'
OUTPUT_PATH = 'datasets/count_product/video/train_detect.mp4'

# Проверка существования файла
if not os.path.exists(VIDEO_PATH):
    print("Файл не найден. Проверьте путь.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

# Задаем новое разрешение кадра
WINDOW_WIDTH, WINDOW_HEIGHT = 480, 640

# Создаем объект для записи видео
output_video = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'),
                               int(cap.get(cv2.CAP_PROP_FPS)), (WINDOW_WIDTH, WINDOW_HEIGHT))

print(f"Запись будет сохранена в: {OUTPUT_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Изменение размера кадра до 480x640
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Обнаружение и трекинг объектов с использованием YOLO
    results = model.track(frame, conf=0.5, persist=True)

    if results[0].boxes.id is not None:  # Проверка наличия треков
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box.tolist())  # Координаты рамки
            cls = int(results[0].boxes.cls[results[0].boxes.id == track_id])

            # Определяем текст и цвет
            label = f"ID: {int(track_id)}"
            color = (255, 0, 0) if cls == 0 else (0, 255, 0)  # Цвет для 'Bag' или 'Product'

            # Отображение рамки и ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Уменьшенная толщина
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)  # Уменьшенный текст

            # Подпись объектов
            if cls == 0:
                cv2.putText(frame, 'Bag', (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif cls == 1:
                cv2.putText(frame, 'Product', (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Запись обработанного кадра
    output_video.write(frame)

    # Отображение кадра в реальном времени
    cv2.imshow('Real-time Bag and Product Tracking', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Видео с трекингом сохранено в: {OUTPUT_PATH}")
