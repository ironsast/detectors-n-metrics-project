from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import time  # Для отслеживания времени обработки

# Загрузка модели YOLOv8
model = YOLO(r"md-density-relay-metrics.pt")

# Новый цвет для выделения: желтый
highlight_color = (0, 255, 255)  # Желтый цвет (BGR)

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
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))  # Ограничиваем значение, чтобы избежать ошибок с вычислением угла
    
    return math.degrees(angle)  # Возвращаем угол в градусах

# Функция для обработки изображения
def process_image(image_path, output_folder, confidence_threshold=0.5):
    start_time = time.time()  # Запуск таймера для времени обработки изображения
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model(image)[0]
    
    # Получение оригинального изображения и результатов
    image_cv = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()

    # Словарь для хранения координат объектов
    objects = {"scalestart": None, "center": None, "scaleend": None, "needle": None}
    colors_map = {}  # Словарь для хранения цветов рамок для каждого объекта

    # Рисование рамок и извлечение координат объектов
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:  # Фильтрация объектов с низкой уверенностью
            continue
        
        class_name = classes_names[int(class_id)]
        color = highlight_color  # Желтый цвет для выделения
        colors_map[class_name] = color  # Сохраняем желтый цвет для данного класса

        if class_name in objects:
            # Записываем координаты центра объекта
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if objects[class_name] is not None:
                print(f"На изображении {image_path} больше одного объекта типа {class_name}, пропускаем.")
                return  # Пропускаем изображение, если найдено больше одного объекта того же типа
            objects[class_name] = (center_x, center_y)
        
        # Рисование рамки и текста с цветом, соответствующим цвету рамки
        x1, y1, x2, y2 = box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)  # Увеличена толщина рамки
        cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)  # Увеличен размер шрифта

    # Проверка, все ли объекты распознаны
    if all(objects.values()):
        # Извлекаем координаты для расчётов
        scalestart, center, scaleend, needle = objects["scalestart"], objects["center"], objects["scaleend"], objects["needle"]
        
        # Вычисление углов
        angle_scalestart_needle = calculate_angle(center, scalestart, needle)
        angle_scalestart_scaleend = 360 - calculate_angle(center, scalestart, scaleend)

        # Вычисляем отношение в процентах
        angle_ratio = (angle_scalestart_needle / angle_scalestart_scaleend) * 100 if angle_scalestart_scaleend != 0 else 0

        # Рисуем линии от центра к остальным объектам, цвет линий и подписей будет желтым
        if scalestart and needle:
            cv2.line(image_cv, center, scalestart, colors_map["scalestart"], 3)
            cv2.line(image_cv, center, needle, colors_map["needle"], 3)
        if scalestart and scaleend:
            cv2.line(image_cv, center, scaleend, colors_map["scaleend"], 3)

        # Рисуем эллипс между center, scalestart и scaleend (желтый цвет)
        if center and scalestart and scaleend:
            axes = (calculate_distance(center, scalestart), calculate_distance(center, scaleend))
            cv2.ellipse(image_cv, center, (int(axes[0]), int(axes[1])), 0, 0, 360, highlight_color, 3)  # Желтый эллипс

        # Конвертируем изображение OpenCV в Pillow для добавления текста в верхнюю часть изображения
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        # Используем шрифт PIL для увеличения текста
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # Увеличиваем размер шрифта
        except IOError:
            font = ImageFont.load_default()  # Если шрифт не найден, используем стандартный шрифт

        # Рисуем текст в верхней части изображения (черным цветом)
        draw.text((10, 30), f"Угол от начала до стрелки: {angle_scalestart_needle:.2f} градусов", fill="red", font=font)
        draw.text((10, 70), f"Угол от начала до конца: {angle_scalestart_scaleend:.2f} градусов", fill="red", font=font)
        draw.text((10, 110), f"Отношение: {angle_ratio:.2f}%", fill="red", font=font)

        processing_time = time.time() - start_time  # Время обработки
        draw.text((10, 150), f"Время: {processing_time:.2f} сек", fill="red", font=font)

        # Преобразуем изображение обратно в формат OpenCV
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Сохранение обработанного изображения
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_yolo{os.path.splitext(image_name)[1]}")
        cv2.imwrite(new_image_path, image_cv)
        print(f"Изображение с распознанными объектами сохранено: {new_image_path}")
    else:
        print(f"Изображение {image_path} не содержит все необходимые объекты, не сохранено.")

# Функция для обработки всех изображений в папке
def process_images_in_folder(input_folder, output_folder, confidence_threshold=0.5):
    # Создание выходной папки, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Поиск всех изображений в папке
    image_paths = glob.glob(os.path.join(input_folder, '*.png')) + glob.glob(os.path.join(input_folder, '*.jpg'))
    
    # Обработка каждого изображения
    for image_path in image_paths:
        process_image(image_path, output_folder, confidence_threshold)

# Пример вызова функции
input_folder = 'upscaled_images'  # Папка с изображениями для обработки
output_folder = 'detected'  # Папка для сохранения обработанных изображений
process_images_in_folder(input_folder, output_folder, confidence_threshold=0.5)
