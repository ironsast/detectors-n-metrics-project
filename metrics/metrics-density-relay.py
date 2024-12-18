from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


model = YOLO(r"metrics-density-relay.pt")
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
            elif line.startswith("Угол от начала до стрелки:"):
                metrics.append(float(line.strip()[:-3].split(": ")[1].replace('°', '')))
            elif line.startswith("Угол от начала до конца:"):
                metrics.append(float(line.strip()[:-3].split(": ")[1].replace('°', '')))
            elif line.startswith("Относительный угол:"):
                metrics.append(float(line.strip()[:-3].split(": ")[1].replace('%', '')))
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

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def calculate_angle(center, point1, point2):
    delta1 = (point1[0] - center[0], point1[1] - center[1])
    delta2 = (point2[0] - center[0], point2[1] - center[1])
    dot_product = delta1[0] * delta2[0] + delta1[1] * delta2[1]
    magnitude1 = math.sqrt(delta1[0] ** 2 + delta1[1] ** 2)
    magnitude2 = math.sqrt(delta2[0] ** 2 + delta2[1] ** 2)
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(angle)

def process_image(image_path, confidence_threshold=0.5):
    image = cv2.imread(image_path)
    results = model(image, verbose=False)[0]
    image_cv = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    confidences = results.boxes.conf.cpu().numpy()
    objects = {"scalestart": None, "center": None, "scaleend": None, "needle": None}
    for class_id, box, conf in zip(classes, boxes, confidences):
        if conf < confidence_threshold:
            continue
        class_name = classes_names[int(class_id)]
        if class_name in objects:
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            objects[class_name] = (center_x, center_y)
    if all(objects.values()):
        for class_id, box, conf in zip(classes, boxes, confidences):
            if conf < confidence_threshold:
                continue
            class_name = classes_names[int(class_id)]
            if class_name in objects:
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                objects[class_name] = (center_x, center_y)
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), highlight_color, 2)
                cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)
                
        scalestart, center, scaleend, needle = objects["scalestart"], objects["center"], objects["scaleend"], objects["needle"]
        cv2.line(image_cv, center, scalestart, highlight_color, 2)
        cv2.line(image_cv, center, scaleend, highlight_color, 2)
        cv2.line(image_cv, center, needle, highlight_color, 2)
        axes = (calculate_distance(center, scalestart), calculate_distance(center, scaleend))
        cv2.ellipse(image_cv, center, (int(axes[0]), int(axes[1])), 0, 0, 360, highlight_color, 2)
        angle_scalestart_needle = calculate_angle(center, scalestart, needle)
        angle_scalestart_scaleend = 360 - calculate_angle(center, scalestart, scaleend)
        angle_ratio = (angle_scalestart_needle / angle_scalestart_scaleend) * 100 if angle_scalestart_scaleend != 0 else 0
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 70), f"Угол от начала до стрелки: {angle_scalestart_needle:.2f}°", fill="red", font=font)
        draw.text((10, 110), f"Угол от начала до конца: {angle_scalestart_scaleend:.2f}°", fill="red", font=font)
        draw.text((10, 150), f"Относительный угол: {angle_ratio:.2f}%", fill="red", font=font)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        new_image_path = os.path.join(os.path.dirname(image_path), f"metrics_{os.path.basename(image_path)}")
        cv2.imwrite(new_image_path, image_cv)
        metrics_data = {
            "Информация для изображения": os.path.basename(image_path),
            "Угол от начала до стрелки": f"{angle_scalestart_needle} px",
            "Угол от начала до конца": f"{angle_scalestart_scaleend} px",
            "Относительный угол": f"{angle_ratio:.2f}"
                        }
        output_metrics_path = os.path.join(os.path.dirname(image_path), "all_metrics.txt")
        append_data_to_file(output_metrics_path, os.path.basename(image_path), metrics_data)
    else:
        if os.path.exists(image_path):
            os.remove(image_path)
        return

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
    
input_folder = '../output_images'
uni_process_images(input_folder, sensor_type='density-relay', confidence_threshold=0.5)
