import cv2
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config("detect_config.json")
input_folder = config['output_folder']

target_size = (1280, 1280)
enhancement_sharpness = 2.0
enhancement_contrast = 1.5
enhancement_brightness = 1.2

def upscale_to_fullhd_and_enhance(input_folder, target_size=(1920, 1080)):
    files = []
    for root, dirs, files_in_dir in os.walk(input_folder):
        for filename in files_in_dir:
            files.append(os.path.join(root, filename))

    for filename in tqdm(files, desc="Улучшение качества изображений", unit=" файл"):
        input_path = filename

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(input_path)
        if image is None:
            continue

        upscaled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))

        upscaled_image = upscaled_image.filter(ImageFilter.DETAIL)
        enhancer = ImageEnhance.Sharpness(upscaled_image)
        upscaled_image = enhancer.enhance(enhancement_sharpness)
        enhancer = ImageEnhance.Contrast(upscaled_image)
        upscaled_image = enhancer.enhance(enhancement_contrast)
        enhancer = ImageEnhance.Brightness(upscaled_image)
        upscaled_image = enhancer.enhance(enhancement_brightness)

        upscaled_image.save(input_path)

upscale_to_fullhd_and_enhance(input_folder, target_size=target_size)
