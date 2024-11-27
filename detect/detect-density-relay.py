import argparse
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Загрузка модели
model = YOLO(r"detect/detect-density-relay.pt")

# Функция для обработки видео и сохранения распознанных областей с объектами
def process_video_and_save_objects(video_path, output_folder, confidence_threshold=0.5, interval_seconds=1):
    # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка при открытии видео {video_path}.")
        return

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров

    # Извлекаем имя видео без расширения для создания папки
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Проверяем, существует ли папка для текущего видео
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)  # Если нет, создаем

    frame_count = 0
    saved_count = 0
    object_count = {}  # Словарь для хранения счетчика объектов по типу

    while True:
        # Чтение кадра
        ret, frame = cap.read()
        if not ret:
            break  # Если кадры закончились, выходим

        # Если текущий кадр не соответствует интервалу, пропускаем его
        if frame_count % int(fps * interval_seconds) != 0:
            frame_count += 1
            continue

        # Детекция объектов на текущем кадре
        results = model(frame)[0]

        # Получение результатов
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
        confidences = results.boxes.conf.cpu().numpy()

        # Сохранение каждого объекта как отдельного изображения
        for i, (class_id, box, conf) in enumerate(zip(classes, boxes, confidences)):
            if conf < confidence_threshold:  # Фильтрация объектов с низкой уверенностью
                continue

            class_name = classes_names[int(class_id)]

            # Игнорируем класс 'number' и заменяем 'gauges' на 'density-relay'
            if class_name == 'number':
                continue
            elif class_name == 'gauges':
                class_name = 'density-relay'

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

    # Освобождение ресурсов
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
