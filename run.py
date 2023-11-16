import subprocess
import time

# Full path to the Python interpreter in your virtual environment
python_path = ".\\.venv\\Scripts\\python.exe"

# Run predict.py
subprocess.Popen([python_path, "predict.py"])

# Wait for a few seconds to ensure the Flask server is running
time.sleep(10)

# Run api.py
subprocess.Popen([python_path, "api.py"])