import subprocess

# Путь к папке с детекторами
detect_folder = 'detect'

# Запуск скрипта detect-density-relay.py
subprocess.run("python detect-density-relay.py", cwd=detect_folder, shell=True)

# Запуск скрипта detect-voltage-transformator.py
subprocess.run("python detect-voltage-transformator.py", cwd=detect_folder, shell=True)

# Запуск скрипта upscale
subprocess.run("python upscale.py", cwd=detect_folder, shell=True)

# Путь к папке с метриками
detect_folder = 'metrics'

# Запуск скрипта metrics-density-relay.py
subprocess.run("python metrics-density-relay.py", cwd=detect_folder, shell=True)

# Запуск скрипта metrics-voltage-transformator.py
subprocess.run("python metrics-voltage-transformator.py", cwd=detect_folder, shell=True)
