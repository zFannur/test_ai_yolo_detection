import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

# Пути и параметры
VIDEO_PATH = "datasets/inactivity_detection/video/train.mp4"       # Входное видео
OUTPUT_PATH = "datasets/inactivity_detection/video/train_detect.mp4" # Выходное видео
MODEL_PATH = "yolo12m.pt"  # Модель YOLO12m для обнаружения людей

# Параметры трекинга и метрика активности
MAX_DISAPPEARED = 50              # Количество кадров, после которых объект считается пропавшим
SPEED_THRESHOLD = 5.0             # Порог смещения за WINDOW_SECONDS (в пикселях)
WINDOW_SECONDS = 1.0              # Интервал для вычисления смещения

class CentroidTracker:
    def __init__(self, maxDisappeared=MAX_DISAPPEARED):
        self.nextObjectID = 0
        self.objects = {}         # objectID -> текущая точка (верхняя центральная точка бокса)
        self.bboxes = {}          # objectID -> последний боксовый прямоугольник (x, y, w, h)
        self.disappeared = {}     # objectID -> количество пропущенных кадров
        self.tracks = {}          # objectID -> список точек для траектории
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.tracks[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]
        del self.tracks[objectID]

    def update(self, rects):
        """
        rects: список обнаруженных боксов для людей в формате (x, y, w, h)
        Положение головы берём как верхняя центральная точка: (x + w/2, y)
        Сопоставление происходит по IoU между новым боксом и последним сохранённым боксом объекта.
        """
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w/2)
            cY = int(y)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for i, centroid in enumerate(inputCentroids):
                self.register(centroid, rects[i])
        else:
            objectIDs = list(self.objects.keys())
            usedDetections = set()
            for objectID in objectIDs:
                best_iou = 0
                best_index = -1
                for i, bbox in enumerate(rects):
                    if i in usedDetections:
                        continue
                    iou = self.compute_iou(self.bboxes[objectID], bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = i
                if best_iou >= 0.3 and best_index != -1:
                    newCentroid = inputCentroids[best_index]
                    newBbox = rects[best_index]
                    self.objects[objectID] = newCentroid
                    self.bboxes[objectID] = newBbox
                    self.tracks[objectID].append(newCentroid)
                    self.disappeared[objectID] = 0
                    usedDetections.add(best_index)
                else:
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            for i, centroid in enumerate(inputCentroids):
                if i not in usedDetections:
                    self.register(centroid, rects[i])
        return self.objects

    def compute_iou(self, boxA, boxB):
        # boxA и boxB в формате (x, y, w, h)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

def process_frame(frame, yolo, tracker, dt):
    # Детектируем людей с помощью YOLO12m (фильтруем класс 0)
    results = yolo(frame, conf=0.3)
    person_rects = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                person_rects.append((x1, y1, w, h))
    objects = tracker.update(person_rects)

    # Количество кадров в окне WINDOW_SECONDS
    frames_in_window = int(WINDOW_SECONDS / dt)
    displacements = {}
    for objectID, centroid in tracker.objects.items():
        pts = tracker.tracks.get(objectID, [])
        if len(pts) >= frames_in_window:
            # Суммарное смещение за WINDOW_SECONDS: расстояние между точкой текущего кадра
            # и точкой WINDOW_SECONDS назад
            disp = math.hypot(pts[-1][0] - pts[-frames_in_window][0],
                              pts[-1][1] - pts[-frames_in_window][1])
        elif len(pts) >= 2:
            # Если недостаточно точек для окна, используем разницу между последними двумя
            disp = math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
        else:
            disp = 0.0
        displacements[objectID] = disp

    # Определяем статус по смещению за WINDOW_SECONDS:
    # если смещение меньше порога SPEED_THRESHOLD, считаем объект неактивным.
    statuses = {objectID: "Working" if disp >= SPEED_THRESHOLD else "Not Working"
                for objectID, disp in displacements.items()}

    # Отрисовка боксов, траекторий и параметров
    for objectID, centroid in tracker.objects.items():
        bbox = tracker.bboxes[objectID]
        x, y, w, h = bbox
        status = statuses.get(objectID, "Not Working")
        box_color = (0, 255, 0) if status == "Working" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, f"ID: {objectID}", (x, y - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.putText(frame, f"Disp: {displacements.get(objectID, 0):.1f}px", (x, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.putText(frame, status, (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        # Отрисовка траектории (последние 10 точек)
        pts = tracker.tracks.get(objectID, [])
        if len(pts) > 10:
            pts = pts[-10:]
        if len(pts) > 1:
            overlay = frame.copy()
            num_pts = len(pts)
            for j in range(num_pts - 1, 0, -1):
                alpha = j / (num_pts - 1)
                color = (int(255 * alpha), 0, 0)
                thickness = 2
                cv2.line(overlay, pts[j - 1], pts[j], color, thickness)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    return frame

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Ошибка открытия видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps  # Время одного кадра
    output_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, output_size)

    yolo = YOLO(MODEL_PATH)
    tracker = CentroidTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        processed_frame = process_frame(frame, yolo, tracker, dt)
        cv2.imshow("Inactivity Detection", processed_frame)
        writer.write(processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
