import subprocess
import sys
import os

# Устанавливаем кодировку для вывода в консоль на UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Папка с входными видео и папка для вывода результатов
input_videos_folder = 'input_videos'
output_folder = 'output_images'

# Проверяем, существует ли папка для вывода. Если нет, создаем ее.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Функция для обработки видео с использованием subprocess
def process_video_with_subprocess(script_path, video_path, output_folder):
    # Формируем команду для subprocess
    command = [
        'python', script_path, video_path, output_folder
    ]

    try:
        # Запуск subprocess и выполнение команды
        result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
        
        # Печать результата выполнения
        print(f"Результат выполнения для {video_path}: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении для {video_path}:")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")

# Проходим по всем видеофайлам в папке input_videos
for video_file in os.listdir(input_videos_folder):
    video_path = os.path.join(input_videos_folder, video_file)

    # Проверяем, является ли файл видеофайлом (можно добавить другие расширения, если нужно)
    if os.path.isfile(video_path) and video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Пример с добавлением '.mkv'
        print(f"Обработка видео: {video_file}")

        # Проходим по всем скриптам в папке detect
        detect_folder = 'detect'
        for script_name in os.listdir(detect_folder):
            script_path = os.path.join(detect_folder, script_name)
                
            # Проверяем, является ли файл Python-скриптом
            if os.path.isfile(script_path) and script_name.endswith('.py'):
                print(f"Запуск скрипта {script_name} для видео {video_file}")

                # Запускаем скрипт с аргументами: путь к видео и папка для вывода результатов
                process_video_with_subprocess(script_path, video_path, output_folder)
