import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Импортируем tqdm для индикатора прогресса

from lib.features.fight.config import (
    SEED, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_FRAMES, IMG_SIZE, NUM_CLASSES, DROPOUT_PROB,
    MODEL_SAVE_DIR, TRAIN_DATA_ROOT
)
from lib.features.fight.dataset import VideoDataset
from lib.features.fight.model_3dcnn import My3DCNN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Трансформации для обучающей выборки (с аугментацией)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Трансформации для валидационной выборки (без аугментации)
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def train_3dcnn():
    # Сбор путей к видеофайлам и меток
    all_video_paths = []
    all_labels = []
    classes = {"noFights": 0, "fights": 1}
    for class_name, label in classes.items():
        folder_path = os.path.join(TRAIN_DATA_ROOT, class_name)
        if not os.path.isdir(folder_path):
            print(f"Папка не найдена: {folder_path}")
            continue
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".mp4", ".avi", ".mpg")):
                all_video_paths.append(os.path.join(folder_path, file_name))
                all_labels.append(label)
    print(f"Найдено видео: {len(all_video_paths)}")

    # Разделение на обучающую и валидационную выборки
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_video_paths, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
    )

    train_dataset = VideoDataset(train_paths, train_labels, num_frames=NUM_FRAMES, transform=train_transforms)
    val_dataset = VideoDataset(val_paths, val_labels, num_frames=NUM_FRAMES, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Инициализация модели
    model = My3DCNN(num_classes=NUM_CLASSES, dropout_prob=DROPOUT_PROB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Используется устройство: {device}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    best_val_loss = float('inf')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, "3dcnn_fight_best.pth")
    final_model_path = os.path.join(MODEL_SAVE_DIR, "3dcnn_fight_final.pth")

    print(f"\n--- Начало обучения ({NUM_EPOCHS} эпох) ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # Индикатор прогресса для обучения
        train_loop = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{NUM_EPOCHS} (обучение)")
        for videos, labels in train_loop:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Обновляем прогресс-бар: можно выводить текущую потерю
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_loop = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{NUM_EPOCHS} (валидация)")
        with torch.no_grad():
            for videos, labels in val_loop:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy*100:.2f}%")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"** Сохранение лучшей модели (Val Loss: {best_val_loss:.4f}) **")

    torch.save(model.state_dict(), final_model_path)
    print(f"Обучение завершено. Финальная модель сохранена: {final_model_path}")
    print(f"Лучшая модель сохранена: {best_model_path}")

if __name__ == "__main__":
    train_3dcnn()
