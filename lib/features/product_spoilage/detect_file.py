import sys
import os
import math
import cv2
from collections import defaultdict, deque
from datetime import datetime
from threading import Thread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
from lib.core.config import MODEL_NAME, DEVICE

# =============================================
# КОНФИГУРАЦИЯ
# =============================================
POSE_MODEL_PATH = 'yolo11s-pose.pt'
BOX_MODEL_PATH = f'models/yolo11_product_spoilage_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'
VIDEO_PATH = "datasets/product_spoilage/video/train2.mp4"
OUTPUT_PATH = "datasets/product_spoilage/video/train_detect_kick.mp4"
LOG_FILE = "events.log"

WINDOW_WIDTH = 480
WINDOW_HEIGHT = 640

# Параметры детекции
POSE_CONF = 0.25
BOX_CONF = 0.3
IOU_THRESH = 0.4

# Параметры событий
DISTANCE_THRESHOLD = 100  # Макс расстояние нога-коробка
FOOT_SPEED_THRESH = 5  # Порог скорости ноги (px/frame)
BOX_ACCEL_THRESH = 5  # Порог ускорения коробки
FALL_SPEED_THRESH = 5  # Порог скорости падения (px/frame)
HISTORY_LENGTH = 5  # Глубина истории для анализа

# Настройки визуализации
COLORS = {
    'left_foot': (0, 0, 255),
    'right_foot': (255, 0, 0),
    'box': (0, 255, 0),
    'fallen_box': (0, 0, 255),
    'speed_text': (255, 255, 0)
}


class KickDetector:
    def __init__(self):
        self.pose_model = YOLO(POSE_MODEL_PATH).to(DEVICE)
        self.box_model = YOLO(BOX_MODEL_PATH).to(DEVICE)

        self.foot_history = deque(maxlen=HISTORY_LENGTH)
        self.box_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
        self.frame_idx = 0

    def process_frame(self, frame):
        self.frame_idx += 1
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        try:
            # Детекция объектов
            pose_results = self.pose_model.predict(
                frame, conf=POSE_CONF, iou=IOU_THRESH,
                device=DEVICE, verbose=False, max_det=10
            )

            box_results = self.box_model.track(
                frame, conf=BOX_CONF, persist=True,
                tracker="botsort.yaml", device=DEVICE, verbose=False
            )

            # Обработка результатов
            limbs = self._process_pose(pose_results, frame)
            boxes = self._process_boxes(box_results, frame)

            # Анализ и визуализация
            self._update_histories(limbs, boxes)
            self._detect_falls(frame, boxes)
            self._analyze_interactions(frame, limbs, boxes)

        except Exception as e:
            print(f"Ошибка обработки кадра: {str(e)}")

        return frame, False

    def _process_pose(self, results, frame):
        limbs = []
        if results[0].keypoints is None:
            return limbs

        for person_id, kpts in enumerate(results[0].keypoints.xy.cpu().numpy()):
            if len(kpts) < 17:
                continue

            left_foot = self._get_valid_point(kpts[15])
            right_foot = self._get_valid_point(kpts[16])

            # Визуализация ног
            self._draw_limb(frame, left_foot, 'left_foot')
            self._draw_limb(frame, right_foot, 'right_foot')

            limbs.append({'left_foot': left_foot, 'right_foot': right_foot})

        return limbs

    def _process_boxes(self, results, frame):
        boxes = {}
        if results[0].boxes.id is None:
            return boxes

        for box, box_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            box_id = int(box_id.item())
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            boxes[box_id] = (cx, cy)

            # Временная визуализация коробок
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['box'], 2)
            cv2.putText(frame, f"Box {box_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['box'], 2)

        return boxes

    def _update_histories(self, limbs, boxes):
        # Обновление истории ног
        current_feet = {}
        for limb in limbs:
            for ft in ['left_foot', 'right_foot']:
                if limb[ft] != (0, 0):
                    current_feet[ft] = limb[ft]
        self.foot_history.append(current_feet)

        # Обновление истории коробок
        for box_id, pos in boxes.items():
            self.box_history[box_id].append(pos)

    def _analyze_interactions(self, frame, limbs, boxes):
        for limb in limbs:
            for foot_type in ['left_foot', 'right_foot']:
                foot_pos = limb[foot_type]
                if foot_pos == (0, 0):
                    continue

                # Поиск ближайшей коробки
                closest_box, min_dist = None, float('inf')
                for box_id, box_pos in boxes.items():
                    dist = math.dist(foot_pos, box_pos)
                    if dist < min_dist:
                        min_dist, closest_box = dist, box_id

                if closest_box is not None and min_dist < DISTANCE_THRESHOLD:
                    foot_speed = self._calculate_limb_speed(foot_type)
                    box_speed = self._calculate_box_speed(closest_box)

                    # Визуализация взаимодействия
                    self._draw_interaction_info(
                        frame,
                        foot_pos,
                        boxes[closest_box],
                        foot_speed,
                        box_speed
                    )

                    if foot_speed > FOOT_SPEED_THRESH and box_speed > BOX_ACCEL_THRESH:
                        self._log_event(f"KICK {foot_type} | Speed: {foot_speed:.1f}")
                        self._draw_alert(frame, "KICK!", foot_pos)

    def _detect_falls(self, frame, boxes):
        for box_id, pos in boxes.items():
            speed = self._calculate_box_speed(box_id)

            # Проверка вертикального движения
            if len(self.box_history[box_id]) >= 2:
                prev_y = self.box_history[box_id][-2][1]
                current_y = pos[1]
                vertical_speed = current_y - prev_y

                if vertical_speed > FALL_SPEED_THRESH:
                    self._log_event(f"FALL Box {box_id} | Speed: {vertical_speed:.1f}")
                    self._draw_fall_alert(frame, pos, box_id)

    def _calculate_limb_speed(self, limb_type):
        if len(self.foot_history) < 2:
            return 0.0

        current = self.foot_history[-1].get(limb_type, (0, 0))
        prev = self.foot_history[-2].get(limb_type, (0, 0))
        return math.hypot(current[0] - prev[0], current[1] - prev[1])

    def _calculate_box_speed(self, box_id):
        history = self.box_history[box_id]
        if len(history) < 2:
            return 0.0

        current = history[-1]
        prev = history[-2]
        return math.hypot(current[0] - prev[0], current[1] - prev[1])

    def _draw_interaction_info(self, frame, foot_pos, box_pos, foot_speed, box_speed):
        try:
            # Линия соединения
            cv2.line(frame, foot_pos, box_pos, (255, 255, 0), 2)

            # Текст скорости
            cv2.putText(frame, f"Foot: {foot_speed:.1f}",
                        (foot_pos[0], foot_pos[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['speed_text'], 2)

            cv2.putText(frame, f"Box: {box_speed:.1f}",
                        (box_pos[0], box_pos[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['speed_text'], 2)

        except Exception as e:
            print(f"Ошибка отрисовки: {str(e)}")

    def _draw_fall_alert(self, frame, pos, box_id):
        x, y = pos
        cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 30), COLORS['fallen_box'], 3)
        cv2.putText(frame, "FALL!", (x - 40, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['fallen_box'], 3)
        cv2.putText(frame, f"ID:{box_id}", (x - 40, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['fallen_box'], 2)

    def _draw_limb(self, frame, pos, limb_type):
        if pos != (0, 0):
            x, y = map(int, pos)
            cv2.circle(frame, (x, y), 8, COLORS[limb_type], -1)
            # cv2.putText(frame, limb_type.split('_')[0].upper(),
            #             (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[limb_type], 2)

    def _draw_alert(self, frame, text, position):
        cv2.putText(frame, text,
                    (position[0] - 50, position[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    def _get_valid_point(self, point):
        try:
            return (int(round(point[0])), int(round(point[1])))
        except:
            return (0, 0)

    def _log_event(self, message):
        def async_log():
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with open(LOG_FILE, "a") as f:
                f.write(f"[{timestamp}] Frame {self.frame_idx}: {message}\n")

        Thread(target=async_log).start()


if __name__ == "__main__":
    detector = KickDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {VIDEO_PATH}")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (WINDOW_WIDTH, WINDOW_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, _ = detector.process_frame(frame)
        out.write(processed_frame)

        cv2.imshow("Kick & Fall Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
