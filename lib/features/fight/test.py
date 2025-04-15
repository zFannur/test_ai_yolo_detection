import os
import subprocess


def reencode_video(input_path, output_path, ffmpeg_path="ffmpeg"):
    """
    Функция перекодирует один видеофайл.

    Аргументы:
      input_path: путь к исходному видеофайлу
      output_path: путь для сохранения перекодированного видео
      ffmpeg_path: путь к исполняемому файлу ffmpeg (по умолчанию ffmpeg)
    """
    command = [
        ffmpeg_path,
        "-i", input_path,
        "-c:v", "libx264",  # Кодек видео: H.264
        "-preset", "veryfast",  # Скоростной режим перекодирования (можно выбрать slow, medium и т.д.)
        "-crf", "23",  # Качество видео: чем ниже значение, тем лучше качество
        "-c:a", "aac",  # Кодек аудио: AAC
        "-b:a", "128k",  # Битрейт аудио: 128 кбит/с
        output_path
    ]

    print(f"Перекодирование файла:\n  Исходный: {input_path}\n  Результат: {output_path}")
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при перекодировании {input_path}:\n{e.stderr}")


def reencode_folder(input_folder, output_folder, target_extensions=(".avi", ".mpg", ".mp4", ".mov", ".mkv"),
                    ffmpeg_path="ffmpeg"):
    """
    Рекурсивно обходит папку input_folder и перекодирует все найденные видеофайлы в output_folder.

    Аргументы:
      input_folder: входная папка с видеофайлами
      output_folder: папка для сохранения перекодированных файлов
      target_extensions: кортеж расширений файлов, которые будут перекодированы
      ffmpeg_path: путь к исполняемому файлу ffmpeg (по умолчанию ffmpeg)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(target_extensions):
                # Полный путь к исходному файлу
                input_path = os.path.join(root, file)
                # Сохраняем ту же структуру папок относительно input_folder
                rel_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, rel_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                # Формируем имя выходного файла с расширением .mp4
                base_name, _ = os.path.splitext(file)
                output_file = base_name + ".mp4"
                output_path = os.path.join(output_subfolder, output_file)

                reencode_video(input_path, output_path, ffmpeg_path)


if __name__ == "__main__":
    # Задайте пути к входной и выходной папкам:
    input_folder = r"C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\datasets\fight\train\noFights"
    output_folder = r"C:\Users\zfann\PycharmProjects\test_ai_yolo_detect\datasets\fight\train\noFights_mp4"

    reencode_folder(input_folder, output_folder)
