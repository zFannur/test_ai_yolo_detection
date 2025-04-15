# lib/detect.py

import cv2
import torch
import numpy as np
import os
from torchvision import transforms
import torch.nn.functional as F

# Импортируем модель
try:
    from lib.features.fight.model_3dcnn import My3DCNN
except ImportError:
    print("Ошибка: Не удалось импортировать My3DCNN.")
    exit()

# Импортируем конфиг
from lib.features.fight.config import (
    NUM_FRAMES, IMG_SIZE, STABLE_THRESHOLD,
    MODEL_SAVE_DIR,
    # Можно импортировать NUM_CLASSES, если нужно
    # и т.д.
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Нормализация (такая же, что применялась при обучении)
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def preprocess_frame_buffer(frames_buffer):
    video_np = np.stack(frames_buffer, axis=0)  # [T, H, W, C]
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()  # -> [T, C, H, W]
    video_tensor = video_tensor / 255.0

    # Нормализация
    normalized_frames = [normalize_transform(frame) for frame in video_tensor]
    normalized_video = torch.stack(normalized_frames)

    # Меняем порядок на [C, T, H, W]
    final_tensor = normalized_video.permute(1, 0, 2, 3)
    return final_tensor

def detect_from_file(video_path, model, device, stable_threshold=STABLE_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: невозможно открыть видеофайл: {video_path}")
        return

    cv2.namedWindow("Fight Detection - Original", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Fight Detection - Model View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fight Detection - Model View", 400, 400)

    frames_buffer = []
    frame_count = 0

    stable_prediction_idx = -1
    stable_confidence = 0.0
    pending_prediction_idx = -1
    pending_count = 0
    raw_prediction_idx = -1
    raw_confidence = 0.0

    classes = {0: "noFight", 1: "fight"}
    colors = {
        0: (0, 255, 0),  # Зеленый
        1: (0, 0, 255)   # Красный
    }
    color_text = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения кадра.")
            break

        # Показываем оригинальный кадр
        display_frame_orig = frame

        # Добавляем кадр в буфер (но перед этим приводим к размеру 112x112)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        frames_buffer.append(resized_frame)
        frame_count += 1

        # Если накопили NUM_FRAMES
        if len(frames_buffer) == NUM_FRAMES:
            video_tensor = preprocess_frame_buffer(frames_buffer).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(video_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                raw_confidence, raw_pred_idx_tensor = torch.max(probabilities, dim=0)
                raw_prediction_idx = raw_pred_idx_tensor.item()
                raw_confidence = raw_confidence.item()

            # Логика стабильности
            if stable_prediction_idx == -1:
                stable_prediction_idx = raw_prediction_idx
                stable_confidence = raw_confidence
                pending_prediction_idx = -1
                pending_count = 0
            else:
                if raw_prediction_idx == stable_prediction_idx:
                    pending_prediction_idx = -1
                    pending_count = 0
                    stable_confidence = raw_confidence
                else:
                    if pending_prediction_idx != raw_prediction_idx:
                        pending_prediction_idx = raw_prediction_idx
                        pending_count = 1
                    else:
                        pending_count += 1
                        if pending_count >= stable_threshold:
                            stable_prediction_idx = pending_prediction_idx
                            stable_confidence = raw_confidence
                            pending_prediction_idx = -1
                            pending_count = 0

            # Берём «центральный» кадр для отображения
            mid_index = NUM_FRAMES // 2
            mid_frame_display = frames_buffer[mid_index]  # RGB
            mid_frame_display_bgr = cv2.cvtColor(mid_frame_display, cv2.COLOR_RGB2BGR)

            # Очистка буфера
            frames_buffer.pop(0)

        # Окно "Model View"
        model_view_h = 400
        model_view_w = 400
        display_frame_processed = np.zeros((model_view_h, model_view_w, 3), dtype=np.uint8)

        # Отображаем центральный кадр, если он есть
        if raw_prediction_idx != -1 and 'mid_frame_display_bgr' in locals():
            resized_for_model = cv2.resize(mid_frame_display_bgr, (224, 224))
            display_frame_processed[0:224, 0:224] = resized_for_model

        # Текст
        text_x = 5
        text_y = 240
        line_step = 20

        if stable_prediction_idx == -1:
            cv2.putText(display_frame_processed, "Stable: Analyzing...",
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 1)
            text_y += line_step
        else:
            stable_text = f"Stable: {classes[stable_prediction_idx]} ({stable_confidence:.2f})"
            stable_color = colors[stable_prediction_idx]
            cv2.putText(display_frame_processed, stable_text,
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stable_color, 2)
            text_y += line_step

        if raw_prediction_idx != -1:
            raw_text = f"Raw: {classes[raw_prediction_idx]} ({raw_confidence:.2f})"
            raw_color = colors[raw_prediction_idx]
            cv2.putText(display_frame_processed, raw_text,
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, raw_color, 2)
            text_y += line_step

        if pending_prediction_idx != -1:
            pending_text = f"Pending: {classes[pending_prediction_idx]} ({pending_count}/{stable_threshold})"
            pending_color = colors[pending_prediction_idx]
            cv2.putText(display_frame_processed, pending_text,
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pending_color, 2)
            text_y += line_step

        cv2.imshow("Fight Detection - Original", display_frame_orig)
        cv2.imshow("Fight Detection - Model View", display_frame_processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Выход по требованию пользователя.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Окна закрыты, выполнение завершено.")


def main():
    print(f"Используется устройство: {DEVICE}")

    # Инициализация модели
    model_input_shape = (3, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
    model = My3DCNN(num_classes=2, input_shape=model_input_shape).to(DEVICE)

    # Путь к финальной (или лучшей) модели
    # Можно взять из config.py, например:
    from lib.features.fight.config import MODEL_SAVE_DIR, TEST_VIDEO_PATH
    model_path = os.path.join(MODEL_SAVE_DIR, "3dcnn_fight_final_ep6_bs32.pth")

    if not os.path.isfile(model_path):
        print(f"Ошибка: не найден файл весов: {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Веса модели успешно загружены из: {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке весов модели: {e}")
        return

    # Файл с видео для теста
    video_path = TEST_VIDEO_PATH
    if not os.path.isfile(video_path):
        print(f"Ошибка: не найден файл видео: {video_path}")
        return

    detect_from_file(video_path, model, DEVICE)


if __name__ == "__main__":
    main()
