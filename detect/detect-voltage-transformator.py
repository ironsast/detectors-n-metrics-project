import argparse
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Загрузка модели один раз
model = YOLO(r"detect/detect-voltage-transformator.pt")

def process_video_and_save_objects(video_path, output_folder, confidence_threshold=0.5, interval_seconds=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка при открытии видео {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Извлекаем имя видео без расширения
    video_name = os.path.basename(video_path).split('.')[0]

    # Создание папки с именем видео, если она не существует
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    frame_count = 0
    saved_count = 0
    object_count = {}  # Словарь для хранения счетчика объектов по типу

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Пропускаем кадры, которые не соответствуют интервалу
        if frame_count % int(fps * interval_seconds) != 0:
            frame_count += 1
            continue

        # Детекция объектов на кадре
        results = model(frame)[0]
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
        confidences = results.boxes.conf.cpu().numpy()

        # Обработка каждого объекта
        for i, (class_id, box, conf) in enumerate(zip(classes, boxes, confidences)):
            if conf < confidence_threshold:
                continue

            class_name = classes_names[int(class_id)]

            # Вырезаем область, соответствующую объекту
            x1, y1, x2, y2 = box
            object_image = frame[y1:y2, x1:x2]  # Вырезка области объекта

            # Увеличиваем счетчик объектов для каждого класса
            if class_name not in object_count:
                object_count[class_name] = 1
            else:
                object_count[class_name] += 1

            # Формируем имя файла с типом объекта и порядковым номером
            object_filename = os.path.join(video_output_folder, f"{class_name}_{object_count[class_name]}.jpg")

            # Сохраняем изображение
            cv2.imwrite(object_filename, object_image)
            saved_count += 1

        frame_count += 1
        print(f"Обработан кадр {frame_count}")

    cap.release()
    print(f"Обработано {frame_count} кадров. Сохранено {saved_count} объектов в папку {video_output_folder}")

# Функция для парсинга аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Обработка видео с детекцией объектов.")
    parser.add_argument("video_path", type=str, help="Путь к видеофайлу для обработки")
    parser.add_argument("output_folder", type=str, help="Папка для сохранения результатов")
    parser.add_argument("--confidence", type=float, default=0.5, help="Порог уверенности для детекции объектов")
    parser.add_argument("--interval", type=float, default=1, help="Интервал между кадрами в секундах")
    return parser.parse_args()

if __name__ == "__main__":
    # Парсим аргументы
    args = parse_args()

    # Вызываем функцию для обработки видео
    process_video_and_save_objects(
        args.video_path,
        args.output_folder,
        confidence_threshold=args.confidence,
        interval_seconds=args.interval
    )
