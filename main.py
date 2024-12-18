import subprocess
import os
import shutil
import time
from datetime import datetime

def run_scripts(folder, scripts):
    for script in scripts:
        subprocess.run(f"python {script}", cwd=folder, shell=True, check=True)

while True:

    detect_folder = 'detect'
    run_scripts(detect_folder, [
        'detect.py',
        'upscale.py'
    ])

    metrics_folder = 'metrics'
    run_scripts(metrics_folder, [
        'metrics-density-relay.py',
        'metrics-voltage-transformator.py'
    ])

    os.system('cls' if os.name == 'nt' else 'clear')

    last_processing_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Последняя проверка завершена в: {last_processing_time}. В ожидании новых данных.")

    time.sleep(15)
