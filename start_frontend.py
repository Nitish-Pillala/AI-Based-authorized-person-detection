"""Entry point to start the Streamlit frontend."""
import subprocess
import sys
import os

if __name__ == "__main__":
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", frontend_path,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
    ])
