# utils/visualization.py
import cv2
from lib.features.product_spoilage.config import config


def draw_box(frame, box_id, box_data):
    try:
        x1, y1, x2, y2 = box_data['coords']
        center = box_data['center']

        # Отрисовка реального bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      config.COLORS["box"], 2)

        # Отрисовка ID
        cv2.putText(frame, f"Box {box_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS["box"], 2)

        # Отрисовка центра
        cv2.circle(frame, center, 4, config.COLORS["box"], -1)

    except KeyError as e:
        print(f"Ошибка отрисовки бокса: отсутствует ключ {str(e)}")
    except Exception as e:
        print(f"Ошибка отрисовки: {str(e)}")


def draw_limb(frame, position, limb_type):
    x, y = position
    cv2.circle(frame, (x, y), 4, config.COLORS[limb_type], -1)


def draw_kick_event(frame, foot_pos, box_pos, speed, box_id):
    # Линия соединения
    cv2.line(frame, foot_pos, box_pos, config.COLORS["speed_text"], 2)

    # Текст скорости
    cv2.putText(frame, f"Kick: {speed:.1f}", (foot_pos[0], foot_pos[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS["speed_text"], 2)

    # Красная рамка
    cv2.rectangle(frame, (box_pos[0] - 30, box_pos[1] - 30),
                  (box_pos[0] + 30, box_pos[1] + 30),
                  config.COLORS["fallen_box"], 3)


def draw_fall_event(frame, position, box_id, speed):
    x, y = position
    cv2.putText(frame, "FALL!", (x - 40, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.COLORS["fallen_box"], 3)
    cv2.putText(frame, f"Speed: {speed:.1f}", (x - 40, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLORS["fallen_box"], 2)