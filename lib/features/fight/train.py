# lib/train.py

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split
import av

# Импортируем конфиг
from lib.features.fight.config import (
    SEED, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_FRAMES, IMG_SIZE, NUM_CLASSES, DROPOUT_PROB,
    MODEL_SAVE_DIR, TRAIN_DATA_ROOT
)

# --- Улучшенный monkey-patch для PyAV ---
try:
    from av import AVError as BaseAVError
    av.AVError = BaseAVError
except ImportError:
    try:
        import av.error
        if not hasattr(av, 'AVError'):
            av.AVError = av.error.OSError
    except (ImportError, AttributeError):
        if not hasattr(av, 'AVError'):
            av.AVError = OSError
# -----------------------------------------

# Импортируем наш датасет и модель
from lib.features.fight.dataset import VideoDataset
from lib.features.fight.model_3dcnn import My3DCNN

# --- Функция для воспроизводимости ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Устанавливаем SEED
set_seed(SEED)

# --- Трансформации ---
# Для обучающей выборки (с аугментацией)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Для валидационной выборки (без аугментации)
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_3dcnn():
    # 1. Собираем пути к файлам и метки
    root_dir = TRAIN_DATA_ROOT  # Берём из конфига
    all_video_paths = []
    all_labels = []
    classes = {"noFights": 0, "fights": 1}

    for class_name, label in classes.items():
        folder_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(folder_path):
            print(f"Предупреждение: Папка не найдена: {folder_path}")
            continue
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".mp4", ".avi", ".mpg")):
                full_path = os.path.join(folder_path, file_name)
                all_video_paths.append(full_path)
                all_labels.append(label)

    if not all_video_paths:
        print(f"Ошибка: Не найдено видеофайлов в {root_dir}")
        return

    print(f"Всего найдено видео: {len(all_video_paths)}")

    # 2. Разделяем на обучающую и валидационную выборки
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_video_paths,
        all_labels,
        test_size=0.2,  # 20% на валидацию
        random_state=SEED,
        stratify=all_labels
    )

    print(f"Обучающих примеров: {len(train_paths)}, Валидационных: {len(val_paths)}")

    # 3. Создаем Dataset'ы
    try:
        train_dataset = VideoDataset(
            file_paths=train_paths,
            labels=train_labels,
            num_frames=NUM_FRAMES,
            transform=train_transforms
        )
        val_dataset = VideoDataset(
            file_paths=val_paths,
            labels=val_labels,
            num_frames=NUM_FRAMES,
            transform=val_transforms
        )
    except Exception as e:
        print(f"Ошибка при создании Dataset: {e}")
        return

    # 4. Dataloader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 5. Инициализируем модель
    input_shape_model = (3, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
    model = My3DCNN(num_classes=NUM_CLASSES, input_shape=input_shape_model, dropout_prob=DROPOUT_PROB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Используется устройство: {device}")

    # 6. Оптимизатор, лосс, scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # 7. Цикл обучения
    best_val_loss = float('inf')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"3dcnn_fight_best_ep{NUM_EPOCHS}_bs{BATCH_SIZE}.pth")
    final_model_path = os.path.join(MODEL_SAVE_DIR, f"3dcnn_fight_final_ep{NUM_EPOCHS}_bs{BATCH_SIZE}.pth")

    print(f"\n--- Начало обучения ({NUM_EPOCHS} эпох) ---")
    for epoch in range(NUM_EPOCHS):
        # --- Обучение ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            if videos is None:
                continue  # если датасет вернул None

            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)  # [B, NUM_CLASSES]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # --- Валидация ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                if videos is None:
                    continue
                videos, labels = videos.to(device), labels.to(device)

                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}%"
        )

        # Шаг scheduler'а
        scheduler.step(avg_val_loss)

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            print(f"** Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model... **")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        print("-" * 80)

    # 8. Сохраняем финальную модель
    torch.save(model.state_dict(), final_model_path)
    print(f"\nОбучение завершено. Финальная модель сохранена: {final_model_path}")
    print(f"Лучшая модель (val_loss={best_val_loss:.4f}) сохранена: {best_model_path}")


if __name__ == "__main__":
    train_3dcnn()
