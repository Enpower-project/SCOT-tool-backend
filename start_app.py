import subprocess
import sys
from pathlib import Path
import dotenv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # ensures venv python is used
dotenv.load_dotenv()
def run(cmd: list[str]):
    print(f"\n> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":

    run([
        PYTHON, "-m", "uvicorn",
        "fast_api.apisrc.service:app",
        "--reload"
    ])

    print("\n✅ Database reset and seeded successfully.")