# python features/smoke/detect_file.py

import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
import cv2
import numpy as np
from lib.core.config import MODEL_NAME, DEVICE, SMOKE_DETECTION_DISTANCE

# Путь к обученной модели
MODEL_PATH = f'models/yolo11_cigarette_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'

# Загрузка обученной модели
model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

# Инициализация видеопотока из файла
VIDEO_PATH = 'datasets/smoke/video/train.mp4'  # Путь к видеофайлу
OUTPUT_PATH = 'datasets/smoke/video/train_detect.mp4'  # Путь для записи обработанного видео

# Проверка существования файла
if not os.path.exists(VIDEO_PATH):
    print("Файл не найден. Проверьте путь.")
    exit()

# Инициализация видеопотока
cap = cv2.VideoCapture(VIDEO_PATH)
# cap = cv2.VideoCapture(2)  # Замените '0' на путь к видеофайлу, если нужно

# Задаем новое разрешение кадра
# WINDOW_WIDTH, WINDOW_HEIGHT = 480, 640
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

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

    # Обнаружение объектов с помощью модели YOLO11
    results = model.predict(frame)

    person_boxes = []
    cigarette_boxes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls == 0:  # Класс 'person'
                person_boxes.append((x1, y1, x2, y2))
                # Рисуем прямоугольник вокруг человека
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2)
            elif cls == 1:  # Класс 'cigarette'
                cigarette_boxes.append((x1, y1, x2, y2))
                # Рисуем прямоугольник вокруг сигареты
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Cigarette', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

    # Проверка расстояния между сигаретой и человеком
    for (x1_cig, y1_cig, x2_cig, y2_cig) in cigarette_boxes:
        cig_center = ((x1_cig + x2_cig) // 2, (y1_cig + y2_cig) // 2)
        for (x1_person, y1_person, x2_person, y2_person) in person_boxes:
            person_center = ((x1_person + x2_person) // 2, (y1_person + y2_person) // 2)
            distance = np.linalg.norm(np.array(cig_center) - np.array(person_center))

            # Если расстояние меньше порога, считаем, что человек курит
            if distance < SMOKE_DETECTION_DISTANCE:  # Настройте порог при необходимости
                cv2.putText(frame, 'Smoking', (x1_person, y1_person - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
                # Обновляем цвет прямоугольника вокруг человека
                cv2.rectangle(frame, (x1_person, y1_person), (x2_person, y2_person), (0, 0, 255), 2)

    # Запись обработанного кадра
    output_video.write(frame)

    # Отображение кадра в реальном времени
    cv2.imshow('Open Product Detection Tracking', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

print(f"Видео с трекингом сохранено в: {OUTPUT_PATH}")
