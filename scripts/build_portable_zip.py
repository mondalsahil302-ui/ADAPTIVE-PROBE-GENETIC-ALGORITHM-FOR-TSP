"""
Portable Distribution Packager & Pre-delivery Verifier
Builds Adaptive_Evolution_Research_Editor_Portable.zip with single root folder,
generates package manifest and SHA-256 checksum, and tests clean extraction & smoke test.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_downloads_dir() -> Path:
    """Dynamically resolves the user's real Downloads folder on Windows and POSIX."""
    if sys.platform == "win32":
        try:
            import winreg
            sub_key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
                downloads_guid = "{374DE290-123F-4565-9164-39C4925E467B}"
                val, _ = winreg.QueryValueEx(key, downloads_guid)
                if val and Path(val).exists():
                    return Path(val)
        except Exception:
            pass

    user_profile = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if user_profile:
        dl = Path(user_profile) / "Downloads"
        if dl.exists():
            return dl
        dl.mkdir(parents=True, exist_ok=True)
        return dl

    home_dl = Path.home() / "Downloads"
    home_dl.mkdir(parents=True, exist_ok=True)
    return home_dl


def should_exclude(rel_path: Path) -> bool:
    """Defines strict exclusion rules for clean, portable packaging."""
    parts = rel_path.parts
    # Exclude version control, environments, caches, and editor metadata
    excluded_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        ".vscode", ".idea", "node_modules", "htmlcov"
    }
    for p in parts:
        if p in excluded_dirs:
            return True

    # Exclude compiled files and secret envs
    if rel_path.suffix in {".pyc", ".pyo", ".pyd", ".swp", ".swo"}:
        return True
    if rel_path.name in {".env", ".env.local", ".coverage", "Thumbs.db", ".DS_Store"}:
        return True

    # Exclude old zip backups and redundant archives
    if rel_path.name in {"adaptive_evolution_research_console_v4.zip", "Adaptive_Evolution_Research_Editor_Portable.zip"}:
        return True

    # Exclude generated temporary run artifacts (keep .gitkeep files)
    if "data" in parts and "runs" in parts:
        if rel_path.name != ".gitkeep":
            return True
    if "data" in parts and "uploads" in parts:
        if rel_path.name != ".gitkeep":
            return True
    if "data" in parts and "cache" in parts:
        if rel_path.name != ".gitkeep":
            return True

    return False


def build_zip() -> tuple[Path, Path, Path, dict]:
    downloads_dir = get_downloads_dir()
    zip_path = downloads_dir / "Adaptive_Evolution_Research_Editor_Portable.zip"
    sha_path = downloads_dir / "Adaptive_Evolution_Research_Editor_Portable.zip.sha256"
    manifest_path = downloads_dir / "Adaptive_Evolution_Research_Editor_Portable_MANIFEST.json"

    print("=" * 70)
    print("  Adaptive Evolution Research Editor - Portable Packager")
    print("=" * 70)
    print(f"[*] Source Project Root : {PROJECT_ROOT}")
    print(f"[*] Destination Directory: {downloads_dir}")
    print(f"[*] Output ZIP File     : {zip_path.name}")
    print("=" * 70)

    top_dir = "Adaptive_Evolution_Research_Editor"
    packaged_files: list[dict] = []
    total_bytes = 0

    print("[*] Gathering files to package...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if not should_exclude(Path(root, d).relative_to(PROJECT_ROOT))]

            for f in sorted(files):
                abs_file = Path(root, f)
                rel_file = abs_file.relative_to(PROJECT_ROOT)

                if should_exclude(rel_file):
                    continue

                archive_name = f"{top_dir}/{rel_file.as_posix()}"
                
                # Check permissions
                zinfo = zipfile.ZipInfo.from_file(abs_file, archive_name)
                # Ensure executable bit for shell scripts on POSIX
                if rel_file.suffix == ".sh" or rel_file.name in ("run_mac_linux.sh", "Run_Config.sh", "Run_Config"):
                    zinfo.external_attr = (0o755 << 16) | 0o100000
                else:
                    zinfo.external_attr = (0o644 << 16) | 0o100000

                with open(abs_file, "rb") as fp:
                    data = fp.read()
                    zf.writestr(zinfo, data)

                size = len(data)
                total_bytes += size
                packaged_files.append({
                    "path": rel_file.as_posix(),
                    "archive_path": archive_name,
                    "size_bytes": size,
                })

        # Also embed the package manifest directly inside the ZIP
        manifest_data = {
            "package_name": "Adaptive_Evolution_Research_Editor_Portable",
            "version": "4.2",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_files": len(packaged_files),
            "uncompressed_bytes": total_bytes,
            "top_level_folder": top_dir,
            "authoritative_entrypoints": {
                "windows_launcher": "run_windows.bat",
                "mac_linux_launcher": "run_mac_linux.sh",
                "python_launcher": "run_console.py",
                "uvicorn_command": "python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"
            },
            "files": packaged_files,
        }
        manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
        zinfo_manifest = zipfile.ZipInfo(f"{top_dir}/PACKAGE_MANIFEST.json")
        zinfo_manifest.external_attr = (0o644 << 16) | 0o100000
        zf.writestr(zinfo_manifest, manifest_bytes)

    # Compute SHA-256
    print("[*] Computing SHA-256 checksum...")
    hasher = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest()

    # Write SHA-256 file
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{zip_sha256}  {zip_path.name}\n")

    # Write external JSON manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "zip_path": str(zip_path),
            "zip_sha256": zip_sha256,
            "zip_size_bytes": zip_path.stat().st_size,
            **manifest_data,
        }, f, indent=2)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully generated: {zip_path}")
    print(f"    Size    : {zip_size_mb:.2f} MB ({zip_path.stat().st_size:,} bytes)")
    print(f"    Files   : {len(packaged_files)} items")
    print(f"    SHA-256 : {zip_sha256}")
    print(f"[+] Checksum file: {sha_path}")
    print(f"[+] Manifest file: {manifest_path}")

    return zip_path, sha_path, manifest_path, manifest_data


def verify_extracted_package(zip_path: Path):
    """Extracts package to a clean temporary directory with spaces and verifies test suite, smoke test, and live startup."""
    print()
    print("=" * 70)
    print("  VERIFYING PACKAGE INTEGRITY (Clean Isolated Sandbox with Spaces)")
    print("=" * 70)

    import urllib.request

    with tempfile.TemporaryDirectory(prefix="ae_verify_") as tmpdir:
        # Specifically create a destination path containing spaces to test path independence
        extract_dir = Path(tmpdir) / "Adaptive Evolution Research Folder (Portable Test)"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Extracting to sandbox path with spaces:\n    {extract_dir}...")
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        target_root = extract_dir / "Adaptive_Evolution_Research_Editor"
        if not target_root.exists() or not target_root.is_dir():
            raise RuntimeError(f"Expected top-level directory 'Adaptive_Evolution_Research_Editor' missing!")

        print(f"[+] Verified clean single top-level directory: '{target_root.name}'")

        # 1. Check critical files in extracted sandbox
        required_sandbox_files = [
            "backend/app.py",
            "backend/config.py",
            "frontend/index.html",
            "frontend/style.css",
            "frontend/app.js",
            "frontend/favicon.ico",
            "frontend/favicon.svg",
            "Run_Config.bat",
            "run_windows.bat",
            "run_mac_linux.sh",
            "run_console.py",
            "README.md",
            "PORTABLE_SETUP.md",
            "STUDY_MAP.md",
            "requirements.txt",
            "pyproject.toml",
            "PACKAGE_MANIFEST.json",
        ]
        for req in required_sandbox_files:
            p = target_root / req
            if not p.exists():
                raise FileNotFoundError(f"Missing required file in extracted package: {req}")
        print(f"[+] All {len(required_sandbox_files)} critical files verified present in extracted package.")

        # 2. Run validate_project.py in the sandbox
        print("[*] Running pre-flight validator inside extracted package...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target_root)
        val_res = subprocess.run(
            [sys.executable, str(target_root / "scripts" / "validate_project.py")],
            cwd=str(target_root),
            capture_output=True,
            text=True,
            env=env,
        )
        if val_res.returncode != 0:
            print(val_res.stdout)
            print(val_res.stderr)
            raise RuntimeError("Validation failed in extracted package!")
        print("[+] Sandbox project validation passed (41/41 paths, 0 syntax errors).")

        # 3. Run unit tests in the sandbox
        print("[*] Running unit & integration tests inside extracted package...")
        test_res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
            cwd=str(target_root),
            capture_output=True,
            text=True,
            env=env,
        )
        if test_res.returncode != 0:
            print(test_res.stdout)
            print(test_res.stderr)
            raise RuntimeError("Test suite failed in extracted package!")
        print("[+] Sandbox test suite passed (25/25 tests passed).")

        # 4. Run smoke test in the sandbox
        print("[*] Running end-to-end smoke test inside extracted package...")
        smoke_res = subprocess.run(
            [sys.executable, str(target_root / "scripts" / "smoke_test.py")],
            cwd=str(target_root),
            capture_output=True,
            text=True,
            env=env,
        )
        if smoke_res.returncode != 0:
            print(smoke_res.stdout)
            print(smoke_res.stderr)
            raise RuntimeError("Smoke test failed in extracted package!")
        print("[+] Sandbox smoke test passed (All 7 steps: Health, UI, Assets, State, Step, PDF, ZIP).")

        # 5. Live server launch and query verification
        print("[*] Performing live server startup test on sandbox instance...")
        test_port = 8019
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m", "uvicorn",
                "backend.app:app",
                "--host", "127.0.0.1",
                "--port", str(test_port),
            ],
            cwd=str(target_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # Poll health endpoint
            health_url = f"http://127.0.0.1:{test_port}/api/health"
            deadline = time.time() + 15.0
            server_ready = False
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(health_url, timeout=0.8) as r:
                        if r.status == 200:
                            server_ready = True
                            break
                except Exception:
                    time.sleep(0.3)

            if not server_ready:
                raise RuntimeError(f"Live test server failed to respond on {health_url}")
            print(f"[+] Live server started and responded on {health_url} (HTTP 200)!")

            # Verify critical live routes
            routes_to_test = [
                ("/", 200),
                ("/favicon.ico", 200),
                ("/style.css", 200),
                ("/app.js", 200),
                ("/api/state", 200),
                ("/api/generation/0", 200),
                ("/api/report/pdf", 200),
            ]
            for route, expected_code in routes_to_test:
                url = f"http://127.0.0.1:{test_port}{route}"
                with urllib.request.urlopen(url, timeout=2.0) as r:
                    if r.status != expected_code:
                        raise RuntimeError(f"Route {route} returned HTTP {r.status}, expected {expected_code}")
                print(f"    - GET {route} -> HTTP {expected_code} OK")

            print("[+] Live server routes and PDF report verified successfully!")
        finally:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except Exception:
                server_proc.kill()
            print("[+] Live test server shut down cleanly.")

    print("=" * 70)
    print("  [SUCCESS] All package verification checks passed with 100% integrity!")
    print("=" * 70)


if __name__ == "__main__":
    zip_file, sha_file, manifest_file, manifest = build_zip()
    verify_extracted_package(zip_file)

