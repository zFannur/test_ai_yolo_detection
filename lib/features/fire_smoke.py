import cv2
import time
from ultralytics import YOLO
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

# Пути к видео
VIDEO_PATH = "datasets/fire_smoke/video/input2.mp4"   # Входное видео (замените на ваш путь)
OUTPUT_PATH = "datasets/fire_smoke/video/output2.mp4"  # Выходной файл

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {VIDEO_PATH}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Проводим патч‑based inference для улучшения обнаружения объектов на кадре
        element_crops = MakeCropsDetectThem(
            image=frame,
            model_path="yolo11m.pt",  # Замените при необходимости на модель для обнаружения дыма/огня
            segment=False,            # Если сегментация не нужна, оставляем False
            shape_x=640,
            shape_y=640,
            overlap_x=25,             # Процент перекрытия по горизонтали
            overlap_y=25,             # Процент перекрытия по вертикали
            conf=0.3,
            iou=0.7,
        )

        # Объединяем детекции с разных патчей и применяем NMS
        result = CombineDetections(element_crops, nms_threshold=0.25)

        # Извлекаем итоговые результаты
        boxes         = result.filtered_boxes      # Список bounding boxes [x_min, y_min, x_max, y_max]
        confidences   = result.filtered_confidences  # Список значений уверенности
        classes_ids   = result.filtered_classes_id   # Список ID классов (числовые)
        classes_names = result.filtered_classes_names  # Список имён классов (строковые)

        # Отрисовка всех обнаруженных объектов и логирование
        for box, conf, cls_name in zip(boxes, confidences, classes_names):
            x1, y1, x2, y2 = map(int, box)
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"{time.strftime('%H:%M:%S')} - Detected {cls_name} with confidence {conf:.2f} at {(x1, y1, x2, y2)}")

        writer.write(frame)
        cv2.imshow("Smoke and Fire Detection (Patch-Based)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
