"""Auto-install dependencies when ComfyUI loads this custom node package."""

import os
import sys
import subprocess

def install():
    aitk_req = os.path.join(os.path.dirname(__file__), "ai-toolkit", "requirements.txt")
    if not os.path.isfile(aitk_req):
        print("[AI Toolkit] requirements.txt not found, skipping auto-install")
        return

    print("[AI Toolkit] Installing ai-toolkit dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-r", aitk_req,
        "--quiet",
    ])
    print("[AI Toolkit] Dependencies installed successfully")

try:
    install()
except Exception as e:
    print(f"[AI Toolkit] Warning: failed to auto-install dependencies: {e}")
    print(f"[AI Toolkit] Please run manually: pip install -r ai-toolkit/requirements.txt")
