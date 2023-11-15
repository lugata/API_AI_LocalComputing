import subprocess

# Menjalankan run.py terlebih dahulu
subprocess.Popen(["python", "predict.py"])

# Menunggu beberapa detik untuk memastikan server Flask sudah berjalan
import time
time.sleep(10)

# Menjalankan test.py
subprocess.Popen(["python", "api.py"])
