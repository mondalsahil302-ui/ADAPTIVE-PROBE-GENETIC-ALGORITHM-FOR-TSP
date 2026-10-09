"""
Convenience launcher for the APGGS 10-Dataset Ablation Study from the repository root.
Delegates directly to APGGS_ABLATION_PROJECT/main.py.
"""
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent / "APGGS_ABLATION_PROJECT"
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from main import main

if __name__ == "__main__":
    main()

