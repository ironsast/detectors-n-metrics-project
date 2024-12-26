import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, config_path, log_path="processed_detect_log.json"):
        self.config = self.load_config(config_path)
        self.log_path = log_path
        self.input_videos_folder = self.config['input_folder']
        self.output_folder = self.config['output_folder']
        self.class_rename_map = self.config.get('class_rename_map', {})
        if not os.path.exists(self.output_folder):
            print(f"Папка {self.output_folder} не существует, создаю её.")
            os.makedirs(self.output_folder)
            # Если папка не существует, очищаем лог
            if os.path.exists(self.log_path):
                print(f"Очищаю лог {self.log_path}.")
                with open(self.log_path, 'w') as f:
                    json.dump({}, f, indent=4)
        self.processed_log = self.load_processed_log()
        self.video_files = [f for f in os.listdir(self.input_videos_folder)
                            if os.path.isfile(os.path.join(self.input_videos_folder, f)) and f.endswith(('.mp4', '.avi', '.mov'))]

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def load_processed_log(self):
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {}

    def save_processed_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.processed_log, f, indent=4)

    def update_processed_log(self, model_path, video_name):
        if model_path not in self.processed_log:
            self.processed_log[model_path] = []
        self.processed_log[model_path].append(video_name)

    def process_video_and_save_objects(self, video_path, model, confidence_threshold=0.5, interval_seconds=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = os.path.basename(video_path).split('.')[0]
        video_output_folder = os.path.join(self.output_folder, video_name)

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
                    if self.class_rename_map and class_name in self.class_rename_map:
                        class_name = self.class_rename_map[class_name]

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

    def process_videos(self):
        # Загрузка моделей и обработка видео
        for model_path in self.config['model_paths']:
            model = YOLO(model_path)  # Загрузка текущей модели
            print(f"Используется модель: {model_path}")
            
            with tqdm(total=len(self.video_files), desc=f"Обработка видео с моделью {os.path.basename(model_path)}", unit=" видео") as overall_pbar:
                for video_file in self.video_files:
                    video_name = os.path.basename(video_file).split('.')[0]

                    # Проверка, было ли видео обработано данной моделью
                    if video_name in self.processed_log.get(model_path, []):
                        print(f"Пропуск видео {video_name}, оно уже обработано моделью {os.path.basename(model_path)}.")
                        overall_pbar.update(1)
                        continue

                    video_path = os.path.join(self.input_videos_folder, video_file)
                    success = self.process_video_and_save_objects(
                        video_path,
                        model=model,
                        confidence_threshold=self.config['confidence_threshold'],
                        interval_seconds=self.config['interval_seconds']
                    )

                    if success:
                        self.update_processed_log(model_path, video_name)
                        self.save_processed_log()
                    
                    overall_pbar.update(1)

# Основная часть
if __name__ == "__main__":
    processor = VideoProcessor(config_path="detect_config.json")
    processor.process_videos()
