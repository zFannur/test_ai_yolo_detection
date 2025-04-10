# README.md

## Обзор

Данный проект направлен на разработку системы, способной определять и анализировать действия на складе с использованием моделей глубокого обучения, таких как YOLO. Основные задачи включают обнаружение курения, отслеживание движения, идентификацию открытых товаров и многое другое.

---

## Функциональность

1.  ✅ Обнаружение курения. (smoke)
2.  ❌ Отслеживание движения внутри склада, манера смотреть по сторонам во время кражи (suspicious_search)
3.  ❌ Определение ношения определенных предметов (например, кепок).
4.  ✅ Подсчет товаров. (items_count)
5.  ✅ Определение, когда человек смотрит прямо в камеру. приоритет (show_camera)
6.  ❌ Обнаружение драк.
7.  ✅ Определение падений. (fall)
8.  ✅ Выявление скрытых объектов. палеты балки (hide_product)
9.  ✅ Контроль вскрытия упаковки. (open_product)
10. ❌ Разметка зон работы.
11. ❌ Интерфейс для обработки событий
12. ✅ подсчитывать время рядом с полкой когда сует руку в стелаж (long_delay_shelf)
13. ✅ падение товара и порча (product_spoilage)
14. ❌ агрессивная жестикуляция
15. ✅ вскрытие пакета внутри склада на пвз (open_product)
16. ✅ насвай принимает (nasvay)
17. ❌ облакачиваются на тележку
18. ❌ падение товара из полки
19. ✅ определение бездействия человека (inactivity_detection)
20. ✅ обнаружение пожара (fire_smoke)
 
---

## Настройка проекта

### Настройка окружения

1. **Создание и активация виртуального окружения:**

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate # Для Linux/MacOS
   .\.venv\Scripts\activate # Для Windows
   ```

2. **Установка CUDA (опционально, для ускорения на видеокартах NVIDIA):**

   - Загрузите с [NVIDIA CUDA](https://developer.nvidia.com/cuda-12-4-0-download-archive) (cuda-12-4-0).
   - После установки проверьте:
     ```bash
     nvcc --version
     ```
   - Установите PyTorch с поддержкой CUDA:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     ```

3. **Установка зависимостей:**

   ```bash
   pip install -r requirements.txt
   ```

### Разделение видео на кадры

1. **Установка FFmpeg:**
   ```bash
   winget install ffmpeg # Для Windows
   ```
2. **Извлечение кадров из видео:**
   ```bash 
   ffmpeg -i train.mp4 -vf fps=1.5 images_train/1_img%04d.png
   ```
   - `train.mp4`: входное видео.
   - `fps=1.5`: частота извлечения кадров.
   - `images_train/1_img%04d.png`: директория и шаблон имен файлов для сохранения.

### Разметка данных

1. **Установка Label Studio:**
   ```bash
   pip install label-studio
   ```
2. **Запуск Label Studio:**
   ```bash
   label-studio start
   ```
3. Откройте инструмент по адресу: `http://localhost:8080`

### Обучение модели

Обучение модели возможно для различных типов данных (например, курение, подсчет товаров и т.д.). Настройте путь к файлу конфигурации и запустите соответствующий скрипт.

Пример для обучения модели курения:

```bash
python features/smoke/train.py
```

Для других типов данных измените путь к конфигурационному файлу в скрипте `train.py`:

```python
import sys
import os

# Добавляем путь к корневой директории
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ultralytics import YOLO
from lib.core.config import MODEL_NAME, EPOCHS, IMG_SIZE, DEVICE


def main():
    # Загрузка предобученной модели YOLO11
    model = YOLO(MODEL_NAME)  # выбираете в файле config.py

    # Перенос модели на устройство
    model.to(DEVICE)

    print(f"Используется модель: {MODEL_NAME}, устройство: {DEVICE}")

    model.train(
        data='open_product.yaml',  # Измените на нужный файл конфигурации
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project='models',
        batch=9,  # Уменьшите до значения, подходящего для вашей системы
        name=f'yolo11_cigarette_detection_{MODEL_NAME.split(".")[0]}',  # измените на нужную имя модели
        pretrained=True,
        device=DEVICE,
        amp=False,
    )


if __name__ == '__main__':
    main()
```

### Для запуска обучения различных типов:

- **Тест**

  ```bash
  python C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\lib\features\test.py
  ```

- **Курение:**

  ```bash
  python features/smoke/train.py
  ```

- **Подсчет товаров:**

  ```bash
  python features/count_product/train.py
  ```

- **Вскрытие упаковки:**

  ```bash
  python features/open_product/train.py
  ```

- **Отслеживание взгляда:**

  ```bash
  python features/look_camera/train.py
  ```
  
- **Отслеживание падения:**

  ```bash
  python features/fall/train.py
  ```

- **Отслеживание порчи имущества:**

  ```bash
  python features/product_spoilage/train.py
  ```

---

## Обнаружение и трекинг

#### Запуск для разных типов обнаружения

1. **Курение:**

   - Реальное время:
     ```bash
     python features/smoke/detect.py
     ```
   - Обработка файлов:
     ```bash
     python features/smoke/detect_file.py
     ```

2. **Подсчет товаров:**

   - Реальное время:
     ```bash
     python features/count_product/detect.py
     ```
   - Обработка файлов:
     ```bash
     python features/count_product/detect_file.py
     ```

3. **Вскрытие упаковки:**

   - Реальное время:
     ```bash
     python features/open_product/detect.py
     ```
   - Обработка файлов:
     ```bash
     python features/open_product/detect_file.py
     ```

4. **Отслеживание взгляда в камеру:**

   - Реальное время:
     ```bash
     python features/look_camera/detect.py
     ```
     
5. **Отслеживание падения:**

   - Реальное время:
     ```bash
     python features/fall/detect.py
     ```
   - Обработка файлов:
     ```bash
     python features/fall/detect_file.py
     ```

6. **Отслеживание порчи имущества:**

   - Реальное время:
     ```bash
     python features/product_spoilage/detect.py
     ```
   - Обработка файлов:
     ```bash
     python C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\lib\features\product_spoilage\main.py
     ```

---

## Дополнительные ресурсы

- готовая модель для обнаружения курения: https://www.kaggle.com/models/sanchris/smoking-detection
- для обнаружения чисел: https://www.kaggle.com/models/abdelrahmanatef01/substitution_board
- обнаруживает животных: https://github.com/jamescdericco/animal-detector
- для обнаружения сиз: https://www.kaggle.com/models/luiscrmartins/yolo11n-trained-ppe-model
- обнаруживает человека и лицо: https://www.kaggle.com/models/luiscrmartins/yolo11s-trained-to-detect-person
- обнаружение животных машин и тд: https://www.kaggle.com/models/agentmorris/megadetector
- оценка скорости движения машин: https://www.kaggle.com/models/pythonistasamurai/yolov10x-supervision-vehicle-speed-estimation
- определяет время прибывания: https://www.kaggle.com/models/pythonistasamurai/yolov10x-supervision-queue-count-model

- Улучшите точность обнаружения с помощью [YOLO Patch-Based Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference).
```pip install patched_yolo_infer```

## Прогресс

1. Составлен список задач и определена архитектура проекта.
2. Проведен сбор и разметка начального датасета.
3. Обучены и протестированы базовые модели.
4. Собраны видеоматериалы для специфических действий (например, курение, обработка товаров).

## Планируемые улучшения

1. Интеграция с интерфейсом обработки событий в реальном времени.
2. Добавление функционала для обнаружения дополнительных действий, таких как драки или падения.
3. Повышение точности обнаружения в сложных условиях (например, при плохом освещении).

---

