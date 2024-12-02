import cv2
from PIL import Image, ImageEnhance, ImageFilter
import os
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Функция для увеличения разрешения до Full HD и улучшения качества изображения
def upscale_to_fullhd_and_enhance(input_folder, target_size=(1920, 1080)):
    # Получаем список всех файлов в папке
    files = []
    for root, dirs, files_in_dir in os.walk(input_folder):
        for filename in files_in_dir:
            files.append(os.path.join(root, filename))

    # Открытие или создание текстового файла для записи имен обработанных файлов
    log_file_path = os.path.join(os.getcwd(), "processed_files.txt")  # Записываем в текущую папку
    with open(log_file_path, "w") as log_file:  # Открытие файла для записи
        # Прогресс-бар для обработки файлов
        for filename in tqdm(files, desc="Улучшение качества изображений", unit=" файл"):
            input_path = filename

            # Проверяем, что это изображение
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Загружаем изображение с помощью OpenCV
            image = cv2.imread(input_path)
            if image is None:
                print(f"Ошибка при загрузке изображения: {filename}")
                continue

            # Увеличиваем разрешение изображения до Full HD
            upscaled_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

            # Преобразуем изображение в формат PIL для дальнейшего улучшения
            upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))

            # Улучшаем резкость, контраст и яркость изображения
            upscaled_image = upscaled_image.filter(ImageFilter.DETAIL)  # Повышение детализации
            enhancer = ImageEnhance.Sharpness(upscaled_image)
            upscaled_image = enhancer.enhance(2.0)  # Повышение резкости
            enhancer = ImageEnhance.Contrast(upscaled_image)
            upscaled_image = enhancer.enhance(1.5)  # Повышение контраста
            enhancer = ImageEnhance.Brightness(upscaled_image)
            upscaled_image = enhancer.enhance(1.2)  # Легкое повышение яркости

            # Сохранение улучшенного изображения, заменяя оригинал
            upscaled_image.save(input_path)

            # Записываем имя обработанного файла в лог
            log_file.write(f"{filename}\n")  # Записываем имя файла в текстовый файл

# Параметры
input_folder = '../output_images'  # Папка с изображениями, которые нужно улучшить

# Вызов функции
upscale_to_fullhd_and_enhance(input_folder, target_size=(1280, 1280))
