import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

# Функция загрузки конфигурации
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Функция загрузки лога обработанных видео
def load_processed_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return {}

# Функция сохранения лога обработанных видео
def save_processed_log(log_path, processed_log):
    with open(log_path, 'w') as f:
        json.dump(processed_log, f, indent=4)

# Функция обновления лога
def update_processed_log(processed_log, model_path, video_name):
    if model_path not in processed_log:
        processed_log[model_path] = []
    processed_log[model_path].append(video_name)

# Основная функция обработки видео
def process_video_and_save_objects(video_path, video_output_folder, model, confidence_threshold=0.5, interval_seconds=1, class_rename_map=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(video_path).split('.')[0]
    video_output_folder = os.path.join(video_output_folder, video_name)

    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    object_count = {}
    saved_count = 0

    with tqdm(total=total_frames, desc=f"Обработка {video_name}", unit=" кадр", leave=False) as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(fps * interval_seconds) != 0:
                frame_count += 1
                continue

            results = model(frame, verbose=False)[0]
            classes_names = results.names
            classes = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
            confidences = results.boxes.conf.cpu().numpy()

            for i, (class_id, box, conf) in enumerate(zip(classes, boxes, confidences)):
                if conf < confidence_threshold:
                    continue

                class_name = classes_names[int(class_id)]

                # Переименование классов по словарю
                if class_rename_map and class_name in class_rename_map:
                    class_name = class_rename_map[class_name]

                x1, y1, x2, y2 = box
                object_image = frame[y1:y2, x1:x2]

                if class_name not in object_count:
                    object_count[class_name] = 1
                else:
                    object_count[class_name] += 1

                object_filename = os.path.join(video_output_folder, f"{class_name}_{object_count[class_name]}.jpg")
                cv2.imwrite(object_filename, object_image)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    return True

# Папки ввода и вывода
config = load_config("detect_config.json")
input_videos_folder = config['input_folder']
output_folder = config['output_folder']

# Словарь для переименования классов
class_rename_map = config.get('class_rename_map', {})

# Лог обработанных видео
log_file = "processed_detect_log.json"
if not os.path.exists(output_folder):
    print(f"Папка {output_folder} не существует, создаю её.")
    os.makedirs(output_folder)

    # Если папка не существует, очищаем лог
    if os.path.exists(log_file):
        print(f"Очищаю лог {log_file}.")
        with open(log_file, 'w') as f:
            json.dump({}, f, indent=4)
processed_log = load_processed_log(log_file)

# Получение списка видеофайлов
video_files = [f for f in os.listdir(input_videos_folder) if os.path.isfile(os.path.join(input_videos_folder, f)) and f.endswith(('.mp4', '.avi', '.mov'))]

# Загрузка моделей из списка и обработка видео
for model_path in config['model_paths']:
    model = YOLO(model_path)  # Загрузка текущей модели
    print(f"Используется модель: {model_path}")
    
    with tqdm(total=len(video_files), desc=f"Обработка видео с моделью {os.path.basename(model_path)}", unit=" видео") as overall_pbar:
        for video_file in video_files:
            video_name = os.path.basename(video_file).split('.')[0]

            # Проверка, было ли видео обработано данной моделью
            if video_name in processed_log.get(model_path, []):
                print(f"Пропуск видео {video_name}, оно уже обработано моделью {os.path.basename(model_path)}.")
                overall_pbar.update(1)
                continue

            video_path = os.path.join(input_videos_folder, video_file)
            success = process_video_and_save_objects(
                video_path,
                output_folder,
                model=model,
                confidence_threshold=config['confidence_threshold'],
                interval_seconds=config['interval_seconds'],
                class_rename_map=class_rename_map
            )

            if success:
                update_processed_log(processed_log, model_path, video_name)
                save_processed_log(log_file, processed_log)
            
            overall_pbar.update(1)
