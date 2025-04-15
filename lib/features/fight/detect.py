import cv2
import torch
import numpy as np
import os
import time
from torchvision import transforms
import torch.nn.functional as F
from lib.features.fight.model_3dcnn import My3DCNN
from lib.features.fight.config import NUM_FRAMES, IMG_SIZE, STABLE_THRESHOLD, MODEL_SAVE_DIR, NUM_CLASSES, TEST_VIDEO_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Нормализация, как при обучении
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def preprocess_frame_buffer(frames):
    """
    Обрабатывает список кадров для подачи в модель.
    Каждый кадр нормализуется и приводится к формату [C, T, H, W].
    """
    video_np = np.stack(frames, axis=0)  # [T, H, W, C]
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = torch.stack([normalize_transform(frame) for frame in video_tensor])
    return video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

def detect_from_file(video_path, model, device, stable_threshold=STABLE_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Не удалось открыть видеофайл.")
        return

    # Определяем целевые размеры для отображения/записи (640x480)
    target_width, target_height = 640, 480
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Формируем путь и имя для сохранения выходного видео
    dir_name = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    output_name = "output_" + base_name
    output_path = os.path.join(dir_name, output_name)

    # Создаем объект VideoWriter с разрешением 640x480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    # Создаем окно отображения (одно окно)
    cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)

    frames_buffer = []
    stable_pred = -1
    pending_pred = -1
    pending_count = 0
    classes = {0: "noFight", 1: "fight"}
    colors = {0: (0, 255, 0), 1: (0, 0, 255)}

    # Для усреднения детекций за 5 секунд
    window_results = []
    detection_window_start = time.time()
    averaged_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Масштабируем исходный кадр до 640x480 без обрезания
        display_frame = cv2.resize(frame, (target_width, target_height))

        # Для подачи в модель используем исходный кадр: перевод в RGB и масштабирование до (IMG_SIZE x IMG_SIZE)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        frames_buffer.append(resized)

        if len(frames_buffer) == NUM_FRAMES:
            video_tensor = preprocess_frame_buffer(frames_buffer).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(video_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                conf, pred = torch.max(probabilities, dim=0)
                current_pred = pred.item()
                confidence = conf.item() * 100  # перевод в проценты
            # Логика быстрой (стабильной) детекции
            if stable_pred == -1:
                stable_pred = current_pred
            else:
                if current_pred != stable_pred:
                    if pending_pred != current_pred:
                        pending_pred = current_pred
                        pending_count = 1
                    else:
                        pending_count += 1
                        if pending_count >= stable_threshold:
                            stable_pred = pending_pred
                            pending_pred = -1
                            pending_count = 0
            frames_buffer.pop(0)
            # Отправляем результат в окно усреднения за 5 секунд
            window_results.append(stable_pred)
            current_time = time.time()
            if current_time - detection_window_start >= 5:
                # Усредняем результаты: если голосов за fight больше, чем за noFight, то итоговая детекция = fight.
                fight_count = window_results.count(1)
                nofight_count = window_results.count(0)
                averaged_result = "fight" if fight_count > nofight_count else "noFight"
                averaged_text = f"Average (5 sec): {averaged_result}"
                # Сброс окна для следующего усреднения
                detection_window_start = current_time
                window_results = []

            # Отображаем надпись с текущим предсказанием и процентом уверенности (уменьшенный шрифт)
            immediate_text = f"Prediction: {classes.get(stable_pred, 'N/A')} ({confidence:.1f}%)"
            cv2.putText(display_frame, immediate_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(stable_pred, (255, 255, 255)), 2)

        # Отображаем отдельно усреднённое определение (более крупным шрифтом, ниже)
        if averaged_text:
            cv2.putText(display_frame, averaged_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Original", display_frame)
        out.write(display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео сохранено: {output_path}")

def main():
    model = My3DCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model_path = os.path.join(MODEL_SAVE_DIR, "3dcnn_fight_final.pth")
    if not os.path.isfile(model_path):
        print("Файл модели не найден.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    detect_from_file(TEST_VIDEO_PATH, model, DEVICE)

if __name__ == "__main__":
    main()
