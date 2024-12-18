# scripts/smoke/detect.py

import sys
import os

import numpy as np

# Добавляем путь к корневой директории
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
import cv2
from scripts.config import MODEL_NAME, EPOCHS, IMG_SIZE, DEVICE, DETECTION_DISTANCE, SMOKE_DETECTION_DISTANCE

# Путь к обученной модели
MODEL_PATH = f'models/yolo11_cigarette_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'

# Загрузка обученной модели
model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

# Инициализация видеопотока
cap = cv2.VideoCapture(2)  # Замените '0' на путь к видеофайлу, если нужно

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # Отображение кадра
    cv2.imshow('Smoking Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
