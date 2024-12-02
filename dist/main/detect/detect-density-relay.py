import cv2
import numpy as np
import os
from ultralytics import YOLO
from tqdm import tqdm  # Импортируем tqdm для прогресс-баров

# Загрузка модели
model = YOLO(r"detect-density-relay.pt")

# Функция для обработки видео и сохранения распознанных областей с объектами
def process_video_and_save_objects(video_path, video_output_folder, confidence_threshold=0.5, interval_seconds=1):
    # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров

    # Извлекаем имя видео без расширения для создания папки
    video_name = os.path.basename(video_path).split('.')[0]

    # Папка для сохранения изображений текущего видео
    video_output_folder = os.path.join(video_output_folder, video_name)

    # Проверяем, существует ли папка для текущего видео, если нет - создаем
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    # Получаем общее количество кадров для отображения прогресса
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    object_count = {}  # Словарь для хранения счетчика объектов по типу
    saved_count = 0

    # Прогресс-бар для отслеживания обработки кадров текущего видео
    with tqdm(total=total_frames, desc=f"Обработка {video_name}", unit=" кадр", leave=False) as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Если кадры закончились, выходим

            # Если текущий кадр не соответствует интервалу, пропускаем его
            if frame_count % int(fps * interval_seconds) != 0:
                frame_count += 1
                continue

            # Детекция объектов на текущем кадре, отключая подробный вывод
            results = model(frame, verbose=False)[0]

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
            pbar.update(1)  # Обновляем прогресс бар для каждого кадра

    # Освобождение ресурсов
    cap.release()

# Папка с видеофайлами и папка для сохранения результатов
input_videos_folder = r"../input_videos"
output_folder = r"../output_images"

# Получаем список видеофайлов
video_files = [f for f in os.listdir(input_videos_folder) if os.path.isfile(os.path.join(input_videos_folder, f)) and f.endswith(('.mp4', '.avi', '.mov'))]

# Прогресс-бар для обработки всех видео
with tqdm(total=len(video_files), desc="Обработка видео density-relay", unit=" видео") as overall_pbar:
    for video_file in video_files:
        video_path = os.path.join(input_videos_folder, video_file)

        # Обработка каждого видео
        process_video_and_save_objects(
            video_path,
            output_folder,
            confidence_threshold=0.5,  # Порог уверенности для детекции
            interval_seconds=1  # Интервал между кадрами
        )
        
        # Обновляем прогресс бар для видео
        overall_pbar.update(1)
