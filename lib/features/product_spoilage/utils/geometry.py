# utils/geometry.py
import math

import numpy as np


def get_valid_point(point):
    try:
        return int(round(point[0])), int(round(point[1]))
    except:
        return (0, 0)


def get_bbox_center(box):
    x1, y1, x2, y2 = map(int, box[:4])
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def calculate_speed(prev_pos, current_pos):
    return math.hypot(current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])


def find_closest_box(point, boxes):
    if not boxes:
        return None

    closest = None
    min_dist = float('inf')

    for box_id, box_data in boxes.items():
        box_pos = box_data['center']  # ← Изменено здесь
        dist = math.dist(point, box_pos)
        if dist < min_dist:
            min_dist = dist
            closest = (box_id, box_data, dist)  # ← Изменено здесь

    return closest


def calculate_limb_speed(history, limb_type, min_frames=2):
    if len(history) < min_frames:
        return 0.0

    current = history[-1].get(limb_type, (0, 0))
    prev = history[-2].get(limb_type, (0, 0))
    return calculate_speed(prev, current)


def calculate_box_speed(history, min_frames=2):
    if len(history) < min_frames:
        return 0.0
    return calculate_speed(history[-2], history[-1])


def calculate_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])
