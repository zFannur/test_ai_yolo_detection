import sys
import os
import math
import cv2
import time
from collections import defaultdict, deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
from lib.core.config import MODEL_NAME

# Конфигурация
VIDEO_PATH = "datasets/product_spoilage/video/train2.mp4"
POSE_MODEL_PATH = 'yolo11s-pose.pt'
BOX_MODEL_PATH = f'models/yolo11_product_spoilage_detection_{MODEL_NAME.split(".")[0]}/weights/best.pt'
HISTORY_LENGTH = 5
LOG_FILE = "kick_events.log"

# Параметры детекции
MIN_KICK_SPEED = 5  # px/frame
MIN_BOX_ACCEL = 10  # px/frame²
MAX_DISTANCE = 70  # px
FPS = 30  # Частота кадров


class KickDetector:
    def __init__(self):
        self.pose_model = YOLO(POSE_MODEL_PATH)
        self.box_model = YOLO(BOX_MODEL_PATH)

        # История объектов
        self.box_history = defaultdict(lambda: {
            'positions': deque(maxlen=HISTORY_LENGTH),
            'velocities': deque(maxlen=HISTORY_LENGTH)
        })

        self.foot_history = defaultdict(lambda: {
            'positions': deque(maxlen=HISTORY_LENGTH)
        })

    def process_frame(self, frame):
        # Детекция объектов
        boxes = self._track_boxes(frame)
        feet = self._detect_feet(frame)

        # Анализ взаимодействий
        self._analyze_interactions(frame, feet, boxes)

        return frame

    def _track_boxes(self, frame):
        boxes = {}
        results = self.box_model.track(
            frame, persist=True,
            tracker="botsort.yaml", conf=0.4, verbose=False
        )

        if results[0].boxes.id is not None:
            for box, box_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                box_id = int(box_id.item())
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                boxes[box_id] = (cx, cy)

                # Обновление истории
                self._update_box_history(box_id, (cx, cy))
                self._draw_box_info(frame, box_id, cx, cy)

        return boxes

    def _update_box_history(self, box_id, pos):
        history = self.box_history[box_id]

        if len(history['positions']) > 0:
            prev_pos = history['positions'][-1]
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            velocity = math.hypot(dx, dy) * FPS
            history['velocities'].append(velocity)

        history['positions'].append(pos)

    def _detect_feet(self, frame):
        feet = []
        results = self.pose_model.predict(
            frame, conf=0.3, iou=0.4, verbose=False
        )

        if results[0].keypoints is not None:
            for kpts in results[0].keypoints.xy.cpu().numpy():
                if len(kpts) < 17: continue

                for i, side in zip([15, 16], ['left', 'right']):
                    x, y = self._get_valid_point(kpts[i])
                    if x > 0 and y > 0:
                        feet.append((x, y))
                        self.foot_history[side].append((x, y))
                        self._draw_foot(frame, (x, y), side)

        return feet

    def _analyze_interactions(self, frame, feet, boxes):
        for foot_pos in feet:
            closest_box = self._find_closest_box(foot_pos, boxes)
            if closest_box is None: continue

            # Расчет параметров
            foot_speed = self._calc_foot_speed(foot_pos)
            box_accel = self._calc_box_accel(closest_box)

            # Логирование для отладки
            print(f"Foot speed: {foot_speed:.1f} | Box accel: {box_accel:.1f}")

            # Проверка условий удара
            if foot_speed > MIN_KICK_SPEED and box_accel > MIN_BOX_ACCEL:
                self._log_event(f"KICK! Speed: {foot_speed:.1f} Accel: {box_accel:.1f}")
                self._draw_kick_alert(frame, foot_pos, boxes[closest_box])

    def _calc_foot_speed(self, foot_pos):
        for side in ['left', 'right']:
            if foot_pos in self.foot_history[side]:
                if len(self.foot_history[side]) < 2: return 0.0
                prev_pos = self.foot_history[side][-2]
                dx = foot_pos[0] - prev_pos[0]
                dy = foot_pos[1] - prev_pos[1]
                return math.hypot(dx, dy) * FPS
        return 0.0

    def _calc_box_accel(self, box_id):
        velocities = self.box_history[box_id]['velocities']
        if len(velocities) < 2: return 0.0
        return (velocities[-1] - velocities[-2]) * FPS

    def _find_closest_box(self, foot_pos, boxes):
        closest_id, min_dist = None, MAX_DISTANCE
        for box_id, pos in boxes.items():
            dist = math.dist(foot_pos, pos)
            if dist < min_dist:
                closest_id, min_dist = box_id, dist
        return closest_id if min_dist <= MAX_DISTANCE else None

    def _draw_box_info(self, frame, box_id, cx, cy):
        velocity = self.box_history[box_id]['velocities'][-1] if len(self.box_history[box_id]['velocities']) > 0 else 0
        accel = self._calc_box_accel(box_id)

        cv2.rectangle(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{box_id}", (cx - 40, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"V:{velocity:.1f} A:{accel:.1f}",
                    (cx - 50, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)

    def _draw_foot(self, frame, pos, side):
        color = (0, 0, 255) if side == 'left' else (255, 0, 0)
        cv2.circle(frame, pos, 8, color, -1)
        cv2.putText(frame, side, (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_kick_alert(self, frame, foot_pos, box_pos):
        cv2.line(frame, foot_pos, box_pos, (0, 255, 255), 2)
        cv2.putText(frame, "KICK!", (foot_pos[0] - 50, foot_pos[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "KICK!", (box_pos[0] - 20, box_pos[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    def _log_event(self, message):
        with open(LOG_FILE, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} | {message}\n")

    def _get_valid_point(self, point):
        try:
            return (int(point[0]), int(point[1]))
        except:
            return (0, 0)


def main():
    detector = KickDetector()
    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        processed_frame = detector.process_frame(frame)

        cv2.imshow("Kick Detection", processed_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()