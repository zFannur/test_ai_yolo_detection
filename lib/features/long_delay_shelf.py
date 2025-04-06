import cv2
import numpy as np
import time
import mediapipe as mp
import math

# Пути и параметры
VIDEO_PATH = "datasets/long_delay_shelf/video/train.mp4"  # Путь к видео
OUTPUT_PATH = "datasets/long_delay_shelf/video/train_detect.mp4"  # Путь для сохранения обработанного видео
LOG_FILE = "hand_in_shelf_events.log"
DELAY_THRESHOLD = 10.0  # Порог времени (в секундах)

# Инициализация MediaPipe Pose и Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


class HandInShelfDetector:
    def __init__(self):
        # Словарь для отслеживания времени, когда условие выполняется для каждой руки
        self.hand_start_time = {"left": None, "right": None}

    def process_frame(self, frame):
        # Поворачиваем изображение вправо (на 90° по часовой стрелке)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        hands_results = hands_detector.process(frame_rgb)

        # Если человек не обнаружен, сбрасываем таймеры и возвращаем кадр без обработки
        if not pose_results.pose_landmarks:
            self.hand_start_time["left"] = None
            self.hand_start_time["right"] = None
            return frame

        landmarks = pose_results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Координаты левой руки (плечо, локоть, запястье: индексы 11, 13, 15)
        left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        left_elbow = (int(landmarks[13].x * w), int(landmarks[13].y * h))
        left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))

        # Координаты правой руки (плечо, локоть, запястье: индексы 12, 14, 16)
        right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        right_elbow = (int(landmarks[14].x * w), int(landmarks[14].y * h))
        right_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))

        # Вычисляем угол в локте для обеих рук
        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Определяем, обнаружена ли кисть с помощью MediaPipe Hands
        left_hand_visible = self._is_hand_visible("left", hands_results)
        right_hand_visible = self._is_hand_visible("right", hands_results)

        # Обрабатываем каждую руку
        self._process_arm(frame, "left", left_angle, left_hand_visible, left_wrist)
        self._process_arm(frame, "right", right_angle, right_hand_visible, right_wrist)

        # Для отладки: отрисовка ключевых точек и углов
        cv2.circle(frame, left_shoulder, 5, (255, 0, 0), -1)
        cv2.circle(frame, left_elbow, 5, (255, 0, 0), -1)
        cv2.circle(frame, left_wrist, 5, (255, 0, 0), -1)
        cv2.putText(frame, f"{int(left_angle)}", left_elbow,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_elbow, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_wrist, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{int(right_angle)}", right_elbow,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def _calculate_angle(self, a, b, c):
        """Вычисляет угол в точке b между точками a, b и c."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        if mag_ba * mag_bc == 0:
            return 0
        angle = math.degrees(math.acos(dot_product / (mag_ba * mag_bc)))
        return angle

    def _is_hand_visible(self, side, hands_results):
        """
        Определяет, видна ли кисть для заданной стороны ("left" или "right")
        с использованием результатов MediaPipe Hands.
        """
        if not hands_results.multi_handedness:
            return False
        for hand_info in hands_results.multi_handedness:
            label = hand_info.classification[0].label  # "Left" или "Right"
            if label.lower() == side:
                return True
        return False

    def _process_arm(self, frame, side, angle, hand_visible, wrist_coord):
        """
        Если рука выпрямлена (угол > 160°) и кисть не обнаружена,
        начинается отсчет времени. Пока задержка меньше DELAY_THRESHOLD секунд,
        выводится обратный отсчёт (оставшиеся секунды) в верхней части экрана.
        При превышении порога выводится предупреждение на английском языке.
        """
        ANGLE_THRESHOLD = 150  # Порог угла для выпрямленной руки
        current_time = time.time()
        if angle > ANGLE_THRESHOLD and not hand_visible:
            if self.hand_start_time[side] is None:
                self.hand_start_time[side] = current_time
            elapsed = current_time - self.hand_start_time[side]
            pos = (10, 30) if side == "left" else (10, 60)
            if elapsed < DELAY_THRESHOLD:
                remaining = int(DELAY_THRESHOLD - elapsed)
                countdown_text = f"Delay {side.capitalize()}: {remaining} sec"
                cv2.putText(frame, countdown_text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                warning_text = f"Warning! {side.capitalize()} hand delayed {DELAY_THRESHOLD}!"
                cv2.putText(frame, warning_text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self._log_event(f"{side.capitalize()} hand delayed for more than {DELAY_THRESHOLD} seconds")
        else:
            self.hand_start_time[side] = None

    def _log_event(self, message):
        """Записывает событие в лог-файл."""
        with open(LOG_FILE, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} | {message}\n")


def main():
    detector = HandInShelfDetector()
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Получаем свойства входного видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Размер кадра после обработки: сначала ресайз до (640, 480), затем поворот на 90° => размер становится (480, 640)
    output_size = (480, 640)

    # Инициализируем VideoWriter для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, output_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ресайз входного кадра (640x480)
        frame = cv2.resize(frame, (640, 480))
        processed_frame = detector.process_frame(frame)
        cv2.imshow("Hand in Shelf Detection", processed_frame)

        # Записываем обработанный кадр в видео
        writer.write(processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
