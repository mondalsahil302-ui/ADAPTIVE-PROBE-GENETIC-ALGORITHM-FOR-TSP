"""
Pre-flight Project Structure and Quality Validator.
Checks file tree integrity, module imports, static delivery, and syntax.
"""
from __future__ import annotations

import os
import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_paths():
    print("[*] Validating critical directory structure and files...")
    required = [
        "backend/__init__.py",
        "backend/app.py",
        "backend/config.py",
        "backend/api/routes.py",
        "backend/schemas/models.py",
        "backend/services/experiment_service.py",
        "backend/optimizer/core.py",
        "backend/optimizer/lkh.py",
        "backend/optimizer/heuristics.py",
        "backend/optimizer/models.py",
        "backend/reports/pdf_generator.py",
        "backend/storage/file_store.py",
        "frontend/index.html",
        "frontend/style.css",
        "frontend/app.js",
        "frontend/assets/css/style.css",
        "frontend/assets/js/app.js",
        "tests/conftest.py",
        "tests/fixtures/demo10.tsp",
        "research/notebooks",
        "research/datasets",
        "scripts/smoke_test.py",
        "scripts/validate_project.py",
        "scripts/benchmark_lkh.py",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        ".python-version",
        ".env.example",
        ".gitignore",
        "render.yaml",
        "README.md",
        "PORTABLE_SETUP.md",
        "STUDY_MAP.md",
        "run_windows.bat",
        "run_mac_linux.sh",
        "artifacts/.gitkeep",
        ".github/workflows/ci-cd.yml",
    ]

    missing = []
    for r in required:
        p = PROJECT_ROOT / r
        if not p.exists():
            missing.append(r)

    if missing:
        print(f"[!] Validation failed. Missing {len(missing)} items:\n" + "\n".join(f"    - {m}" for m in missing))
        return False
    print(f"[+] All {len(required)} critical files and paths are present.")
    return True


def check_syntax():
    print("[*] Compiling all Python source files...")
    failed = []
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if any(skip in py_file.parts for skip in (".venv", "venv", "__pycache__", ".pytest_cache")):
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            failed.append((py_file, str(e)))

    if failed:
        print(f"[!] Syntax check failed on {len(failed)} files:")
        for f, err in failed:
            print(f"    {f}: {err}")
        return False
    print("[+] All Python source files compiled with 0 syntax errors.")
    return True


def check_imports():
    print("[*] Validating backend imports...")
    try:
        import backend.app
        import backend.config
        import backend.optimizer
        import backend.reports
        import backend.storage
        import backend.services
        import backend.api
        print("[+] All backend modules imported successfully.")
        return True
    except Exception as e:
        print(f"[!] Import failure: {e}")
        return False


def main():
    print("=" * 65)
    print("  Adaptive Evolution Research Console - Project Validator")
    print("=" * 65)
    ok1 = check_paths()
    ok2 = check_syntax()
    ok3 = check_imports()
    print("=" * 65)
    if ok1 and ok2 and ok3:
        print("  [SUCCESS] Repository structure is 100% valid and production-ready!")
        sys.exit(0)
    else:
        print("  [FAILURE] Some validation checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
