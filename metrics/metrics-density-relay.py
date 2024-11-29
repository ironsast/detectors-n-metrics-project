from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# Загрузка модели YOLOv8
model = YOLO(r"metrics-density-relay.pt")
highlight_color = (0, 255, 255)

#Универсальные функции

import os
import glob
from tqdm import tqdm

# Универсальная функция для обработки всех изображений в папке и её подкаталогах (рекурсивно)
def uni_process_images(input_folder, sensor_type, confidence_threshold=0.5):
    # Получение списка всех изображений с типом датчика в названии и без суффикса "_processed"
    image_paths = []
    
    # Поиск всех изображений в подкаталогах
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if sensor_type.lower() in file.lower() and '_processed' not in file.lower() and file.lower().endswith(('png', 'jpg')):
                image_paths.append(os.path.join(root, file))
    
    # Обработка каждого изображения с прогресс-баром
    for image_path in tqdm(image_paths, desc=f"Обработка изображений для {sensor_type}", unit=" изображение"):
        process_image(image_path, confidence_threshold)


#Уникальные функции

# Функция для вычисления расстояния между двумя точками
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Функция для вычисления угла между двумя точками относительно центра (в градусах)
def calculate_angle(center, point1, point2):
    delta1 = (point1[0] - center[0], point1[1] - center[1])
    delta2 = (point2[0] - center[0], point2[1] - center[1])
    
    dot_product = delta1[0] * delta2[0] + delta1[1] * delta2[1]
    magnitude1 = math.sqrt(delta1[0] ** 2 + delta1[1] ** 2)
    magnitude2 = math.sqrt(delta2[0] ** 2 + delta2[1] ** 2)
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))  # Ограничиваем значение, чтобы избежать ошибок
    return math.degrees(angle)  # Возвращаем угол в градусах

# Функция для обработки изображения
def process_image(image_path, confidence_threshold=0.5):
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model(image, verbose=False)[0]
    
    # Получение оригинального изображения и результатов
    image_cv = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()

    # Словарь для хранения координат объектов
    objects = {"scalestart": None, "center": None, "scaleend": None, "needle": None}

    # Извлечение координат объектов
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:
            continue
        
        class_name = classes_names[int(class_id)]
        if class_name in objects:
            # Записываем координаты центра объекта
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            objects[class_name] = (center_x, center_y)
            

    # Проверка, все ли объекты распознаны
    if all(objects.values()):
        for class_id, box, conf in zip(classes, boxes, confidences):
            if class_name in objects:
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), highlight_color, 2)
                cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
        
        scalestart, center, scaleend, needle = objects["scalestart"], objects["center"], objects["scaleend"], objects["needle"]

        # Рисуем линии от центра к объектам
        cv2.line(image_cv, center, scalestart, highlight_color, 2)
        cv2.line(image_cv, center, scaleend, highlight_color, 2)
        cv2.line(image_cv, center, needle, highlight_color, 2)

        # Рисуем эллипс вокруг объектов
        axes = (calculate_distance(center, scalestart), calculate_distance(center, scaleend))
        cv2.ellipse(image_cv, center, (int(axes[0]), int(axes[1])), 0, 0, 360, highlight_color, 2)

        # Вычисление углов
        angle_scalestart_needle = calculate_angle(center, scalestart, needle)
        angle_scalestart_scaleend = 360 - calculate_angle(center, scalestart, scaleend)
        angle_ratio = (angle_scalestart_needle / angle_scalestart_scaleend) * 100 if angle_scalestart_scaleend != 0 else 0

        # Конвертируем изображение в Pillow для добавления текста
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()

        # Добавляем текстовую информацию
        draw.text((10, 10), f"Angle start to needle: {angle_scalestart_needle:.2f}°", fill="red", font=font)
        draw.text((10, 40), f"Angle start to end: {angle_scalestart_scaleend:.2f}°", fill="red", font=font)
        draw.text((10, 70), f"Angle ratio: {angle_ratio:.2f}%", fill="red", font=font)

        # Преобразуем изображение обратно в формат OpenCV
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        # Сохранение обработанного изображения с суффиксом "_processed"
        new_image_path = os.path.splitext(image_path)[0] + "_processed" + os.path.splitext(image_path)[1]
        cv2.imwrite(new_image_path, image_cv)
    else:
        os.remove(image_path)
        return

    



# Пример вызова функции
input_folder = '../output_images'  # Входная папка с изображениями
uni_process_images(input_folder, sensor_type='density-relay', confidence_threshold=0.5)
