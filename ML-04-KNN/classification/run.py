import subprocess
from pathlib import Path

# รายชื่อไฟล์ที่ต้องการสั่งรันตามลำดับ
files_to_run = [
    "data_loader.py",
    "main.py",
    "evaluate.py",
    "knn_tf.py",
]

for file in files_to_run:
    file_path = Path(__file__).parent / file
    if file_path.exists():
        print(f"\n================ Running {file} ================")
        subprocess.run(["python", str(file_path)])
    else:
        print(f"File not found: {file}")