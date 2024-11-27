from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Импортируем PIL для работы с текстом

# Загрузка модели YOLOv8
model = YOLO(r"md-voltage-transformator-metrics.pt")

# Список цветов для различных классов
colors = [(0, 255, 255)]  # Используем только желтый цвет для рамок

# Функция для рисования параллелограмма
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

    return minpos_y, maxpos_y, level_center_x  # Возвращаем нужные координаты для расчета расстояний

# Функция для обработки изображения
def process_image(image_path, output_folder, confidence_threshold=0.5):
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model(image)[0]
    
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()

    # Словарь для хранения координат объектов
    centers = {"minpos": None, "maxpos": None, "level": None}
    
    # Рисование рамок и извлечение центров объектов
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:  # Фильтрация объектов с низкой уверенностью
            continue
        
        class_name = classes_names[int(class_id)]
        color = colors[0]  # Все рамки будут желтые

        # Инициализация переменных для центра объекта
        center_x, center_y = None, None

        # Если объект относится к `minpos`, `maxpos` или `level`, записываем его центр
        if class_name in centers:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centers[class_name] = (center_x, center_y)

        # Рисуем рамки вокруг объектов и добавляем подпись с дополнительной информацией
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Добавляем текстовую информацию (желтый цвет)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Проверка, что все необходимые объекты найдены
    if centers["minpos"] and centers["maxpos"] and centers["level"]:
        # Рисуем новый параллелограмм с учетом объектов minpos, maxpos и level
        # Находим индекс для класса "level" с использованием словаря
        level_index = list(classes_names.values()).index("level")
        level_box = boxes[classes == level_index][0]  # Получаем координаты level
        minpos_y, maxpos_y, level_center_x = draw_parallelogram(image, centers["minpos"], centers["maxpos"], level_box)

        # Вычисление расстояний между центрами
        distance_minmax = abs(centers["minpos"][1] - centers["maxpos"][1])  # Расстояние между центрами minpos и maxpos
        distance_minlevel = abs(centers["minpos"][1] - centers["level"][1])  # Расстояние между центрами minpos и level
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

    # Сохранение обработанного изображения
    image_name = os.path.basename(image_path)
    new_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_yolo{os.path.splitext(image_name)[1]}")
    cv2.imwrite(new_image_path, image)
    print(f"Изображение с распознанными объектами сохранено: {new_image_path}")

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
input_folder = 'output_images'  # Папка с изображениями для обработки
output_folder = 'detected'  # Папка для сохранения обработанных изображений
process_images_in_folder(input_folder, output_folder, confidence_threshold=0.5)
