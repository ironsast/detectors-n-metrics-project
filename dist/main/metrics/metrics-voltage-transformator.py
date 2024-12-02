import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np
import time  
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import shutil
import math

model = YOLO(r"metrics-voltage-transformator.pt")
highlight_color = (0, 255, 255)

def append_data_to_file(file_path, image_name, data_dict):
    with open(file_path, 'a') as file: 
        file.write(f"Изображение: {image_name}\n")
        for key, value in data_dict.items():
            file.write(f"{key}: {value}\n")
        file.write("=" * 40 + "\n") 

def extract_metrics_from_file(file_path):
    metrics = []
    image_name = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Изображение:"):
                if image_name is not None and len(metrics) > 0:
                    yield image_name, metrics
                image_name = line.strip().split(": ")[1]
                metrics = []
            elif line.startswith("Расстояние от начала до конца:"):
                metrics.append(float(line.strip().split(": ")[1].replace(' px', '')))
            elif line.startswith("Расстояние от начала до уровня:"):
                metrics.append(float(line.strip().split(": ")[1].replace(' px', '')))
            elif line.startswith("Относительное расстояние:"):
                metrics.append(float(line.strip().split(": ")[1].replace('%', '')))
        if image_name is not None and len(metrics) > 0:
            yield image_name, metrics

def calculate_ev_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def choose_best_image(txt_file):
    all_metrics = list(extract_metrics_from_file(txt_file))
    avg_metrics = [0, 0, 0]
    for image_name, metrics in all_metrics:
        for i in range(len(metrics)):
            avg_metrics[i] += metrics[i]
    avg_metrics = [x / len(all_metrics) for x in avg_metrics]
    best_image = None
    min_distance = float('inf')
    for image_name, metrics in all_metrics:
        distance = calculate_ev_distance(metrics, avg_metrics)
        if distance < min_distance:
            min_distance = distance
            best_image = image_name

    return best_image

def draw_parallelogram(image, minpos, maxpos, level_box):
    x1, y1, x2, y2 = level_box
    minpos_x, minpos_y = minpos
    maxpos_x, maxpos_y = maxpos
    top_left = (x1, maxpos_y)  
    top_right = (x2, maxpos_y)  
    bottom_left = (x1, minpos_y)  
    bottom_right = (x2, minpos_y) 
    cv2.line(image, top_left, top_right, (0, 255, 255), 2)  
    cv2.line(image, bottom_left, bottom_right, (0, 255, 255), 2)  
    cv2.line(image, top_left, bottom_left, (0, 255, 255), 2)  
    cv2.line(image, top_right, bottom_right, (0, 255, 255), 2)  
    level_center_x = (x1 + x2) // 2
    cv2.line(image, (level_center_x, minpos_y), (level_center_x, maxpos_y), (0, 255, 255), 2)
    return minpos_y, maxpos_y, level_center_x  

def process_image(image_path, confidence_threshold=0.5):
    image = cv2.imread(image_path)
    results = model(image, verbose=False)[0] 
    image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()
    objects = {"minpos": None, "maxpos": None, "level": None, "min": None, "max": None}
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:  
            continue
        class_name = classes_names[int(class_id)]
        color = highlight_color  
        center_x, center_y = None, None
        if class_name in objects:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            objects[class_name] = (center_x, center_y)

    if all(objects.values()):
        for class_id, box, conf in zip(classes, boxes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
            if class_name in objects:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
        level_index = list(classes_names.values()).index("level")
        level_box = boxes[classes == level_index][0]
        minpos_y, maxpos_y, level_center_x = draw_parallelogram(image, objects["minpos"], objects["maxpos"], level_box)
        distance_minmax = abs(objects["minpos"][1] - objects["maxpos"][1])  
        distance_minlevel = abs(objects["minpos"][1] - objects["level"][1])  
        ratio = distance_minlevel / distance_minmax if distance_minmax != 0 else 0  
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text((10, 70), f"Расстояние от начала до конца: {distance_minmax} px", font=font, fill=(255, 0, 0))  # Красный текст
        draw.text((10, 110), f"Расстояние от начала до уровня: {distance_minlevel} px", font=font, fill=(255, 0, 0))  # Красный текст
        draw.text((10, 150), f"Относительное расстояние: {ratio:.2f}", font=font, fill=(255, 0, 0))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        new_image_path = os.path.join(os.path.dirname(image_path), f"metrics_{os.path.basename(image_path)}")
        cv2.imwrite(new_image_path, image)
        metrics_data = {
            "Информация для изображения": os.path.basename(image_path),
            "Расстояние от начала до конца": f"{distance_minmax} px",
            "Расстояние от начала до уровня": f"{distance_minlevel} px",
            "Относительное расстояние": f"{ratio:.2f}"
                        }
        output_metrics_path = os.path.join(os.path.dirname(image_path), "all_metrics.txt")
        append_data_to_file(output_metrics_path, os.path.basename(image_path), metrics_data)
    else:
        if os.path.exists(image_path):
            os.remove(image_path)


def uni_process_images(input_folder, sensor_type, confidence_threshold=0.5):
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if sensor_type.lower() in file.lower() and 'metrics_' not in file.lower() and file.lower().endswith(('png', 'jpg')):
                image_paths.append(os.path.join(root, file))
                
    for image_path in tqdm(image_paths, desc=f"Обработка изображений для {sensor_type}", unit=" изображение"):
        process_image(image_path, confidence_threshold)
        
    txt_file_paths = []
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file == "all_metrics.txt":
                txt_file_paths.append(os.path.join(root, file))
            if sensor_type.lower() in file.lower() and file.lower().endswith(('png', 'jpg')):
                image_paths.append(os.path.join(root, file))
    for txt_file_path in txt_file_paths:
        if list(extract_metrics_from_file(txt_file_path)):
            best_image = choose_best_image(txt_file_path)
            txt_file_dir = os.path.dirname(txt_file_path)
            for image_path in image_paths:
                image_dir = os.path.dirname(image_path)
                if image_dir == txt_file_dir:
                    if os.path.basename(image_path) != best_image and os.path.basename(image_path) != "metrics_" + best_image and os.path.exists(image_path):
                        os.remove(image_path)
    for root, dirs, files in os.walk(input_folder):
        if not files and not dirs:
            metrics_file_path = os.path.join(root, "all_metrics.txt")
            if not os.path.exists(metrics_file_path):
                with open(metrics_file_path, 'w') as file:
                    file.write("Ничего не обнаружено.\n")

input_folder = r'../output_images'
uni_process_images(input_folder, sensor_type="voltage-transformator", confidence_threshold=0.5)  
