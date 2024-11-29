import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np
import time  # Для отслеживания времени обработки
from tqdm import tqdm  # Импортируем прогресс-бар для отображения хода обработки
from PIL import Image, ImageDraw, ImageFont
import shutil  # Для копирования исходных изображений

# Загружаем модель для детекции
model = YOLO(r"metrics-voltage-transformator.pt")
highlight_color = (0, 255, 255)  # Цвет рамок для объектов

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

def save_data_to_file(file_path, data_dict):
    try:
        with open(file_path, 'w') as file:
            for key, value in data_dict.items():
                file.write(f"{key}: {value}\n")
            file.write("=" * 40 + "\n")  # Разделитель для ясности
    except Exception as e:

# Функция для рисования параллелограмма между объектами
def draw_parallelogram(image, minpos, maxpos, level_box):
    # Получаем координаты области level
    x1, y1, x2, y2 = level_box
    
    # Центры объектов minpos и maxpos
    minpos_x, minpos_y = minpos
    maxpos_x, maxpos_y = maxpos
    
    # Левый верхний угол будет на уровне верхней границы maxpos
    top_left = (x1, maxpos_y)  # Левый верхний угол на верхней границе maxpos
    top_right = (x2, maxpos_y)  # Правый верхний угол на верхней границе maxpos
    
    # Нижняя линия будет на уровне нижней границы minpos
    bottom_left = (x1, minpos_y)  # Левый нижний угол на уровне minpos
    bottom_right = (x2, minpos_y)  # Правый нижний угол на уровне minpos
    
    # Рисуем верхнюю и нижнюю стороны параллелограмма
    cv2.line(image, top_left, top_right, (0, 255, 255), 2)  # Верхняя линия через верхнюю границу maxpos
    cv2.line(image, bottom_left, bottom_right, (0, 255, 255), 2)  # Нижняя линия через нижнюю границу minpos

    # Рисуем боковые стороны параллелограмма
    cv2.line(image, top_left, bottom_left, (0, 255, 255), 2)  # Левая боковая линия
    cv2.line(image, top_right, bottom_right, (0, 255, 255), 2)  # Правая боковая линия

    # Центр области level (по горизонтали)
    level_center_x = (x1 + x2) // 2

    # Добавление линии снизу вверх, проходящей через центр level
    cv2.line(image, (level_center_x, minpos_y), (level_center_x, maxpos_y), (0, 255, 255), 2)

    # Возвращаем необходимые координаты для дальнейших расчетов
    return minpos_y, maxpos_y, level_center_x  

# Функция для обработки изображения
def process_image(image_path, confidence_threshold=0.5):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Обработка изображения с помощью модели YOLO
    results = model(image, verbose=False)[0]  # Отключаем вывод YOLO
    
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()

    # Словарь для хранения координат объектов
    objects = {"minpos": None, "maxpos": None, "level": None, "min": None, "max": None}
    
    # Рисование рамок и извлечение центров объектов
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:  # Фильтрация объектов с низкой уверенностью
            continue
        
        class_name = classes_names[int(class_id)]
        color = highlight_color  # Все рамки будут желтые

        # Инициализация переменных для центра объекта
        center_x, center_y = None, None

        # Если объект относится к minpos, maxpos или level, записываем его центр
        if class_name in objects:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            objects[class_name] = (center_x, center_y)
        
    # Проверка, что все необходимые объекты найдены
    if all(objects.values()):
        for class_id, box, conf in zip(classes, boxes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # Добавляем текстовую информацию (желтый цвет)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
            if class_name in objects:
                # Рисуем рамки вокруг объектов
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Добавляем текстовую информацию (желтый цвет)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
        
        # Рисуем новый параллелограмм с учетом объектов minpos, maxpos и level
        level_index = list(classes_names.values()).index("level")
        level_box = boxes[classes == level_index][0]  # Получаем координаты level
        minpos_y, maxpos_y, level_center_x = draw_parallelogram(image, objects["minpos"], objects["maxpos"], level_box)

        # Вычисление расстояний между центрами
        distance_minmax = abs(objects["minpos"][1] - objects["maxpos"][1])  # Расстояние между центрами minpos и maxpos
        distance_minlevel = abs(objects["minpos"][1] - objects["level"][1])  # Расстояние между центрами minpos и level
        ratio = distance_minlevel / distance_minmax if distance_minmax != 0 else 0  # Отношение расстояний

        # Создаем объект Image для добавления текста с использованием PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Задаем шрифт, который поддерживает кириллицу
        font = ImageFont.truetype("arial.ttf", 20)

        # Добавление текста с информацией в левый верхний угол
        draw.text((10, 70), f"Расстояние min-max: {distance_minmax} px", font=font, fill=(255, 0, 0))  # Красный текст
        draw.text((10, 110), f"Расстояние min-level: {distance_minlevel} px", font=font, fill=(255, 0, 0))  # Красный текст
        draw.text((10, 150), f"Отношение: {ratio:.2f}", font=font, fill=(255, 0, 0))  # Красный текст

        # Преобразуем обратно в формат OpenCV
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # Сохранение обработанного изображения с префиксом "metrics_"
        processed_image_path = os.path.join(os.path.dirname(image_path), f"metrics_{os.path.basename(image_path)}")
        cv2.imwrite(processed_image_path, image)  # Сохраняем обработанное изображение
        metrics_data = {
            "Информация для изображения": os.path.basename(image_path),
            "Расстояние min-max": f"{distance_minmax} px",
            "Расстояние min-level": f"{distance_minlevel} px",
            "Отношение": f"{ratio:.2f}"
                        }
        save_data_to_file(f"{os.path.splitext(image_path)[0]}_metrics.txt", metrics_data)
    else:
        os.remove(image_path)  # Удаляем изображение, если не найдены все объекты

    

# Пример вызова функции
input_folder = r'../output_images'  # Папка с изображениями для обработки
uni_process_images(input_folder, sensor_type="voltage-transformator", confidence_threshold=0.5)  # Обработка изображений
