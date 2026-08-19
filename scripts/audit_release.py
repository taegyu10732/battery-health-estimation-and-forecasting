"""Fail when release scripts or files contain common publication hazards."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SOURCE = {
    "pinned CUDA device": re.compile(r"CUDA_VISIBLE_DEVICES|cuda:\d+"),
    "GPU shell probe": re.compile(r"nvidia-smi", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\"),
    "Colab drive path": re.compile(r"/content/drive/"),
    "IPython runtime call": re.compile(r"get_ipython\s*\("),
    "Jupyter display call": re.compile(r"\bdisplay\s*\("),
}
FORBIDDEN_SUFFIXES = {".mat", ".pt", ".pth", ".ckpt", ".pkl", ".joblib", ".onnx"}
SCRIPT_ROOTS = (ROOT / "workflows", ROOT / "research_scripts")


def audit_script(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    for label, pattern in FORBIDDEN_SOURCE.items():
        if pattern.search(source):
            errors.append(f"{path.relative_to(ROOT)} contains {label}")
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        errors.append(f"{path.relative_to(ROOT)} is not valid Python: {exc.msg}")
    return errors


def main() -> int:
    errors: list[str] = []
    scripts = sorted(path for directory in SCRIPT_ROOTS for path in directory.rglob("*.py"))
    if len(scripts) != 39:
        errors.append(f"Expected 39 public scripts, found {len(scripts)}")
    for path in scripts:
        errors.extend(audit_script(path))

    notebooks = [path for path in ROOT.rglob("*.ipynb") if ".venv" not in path.parts]
    for path in notebooks:
        errors.append(f"Notebook remains in Python-only release: {path.relative_to(ROOT)}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".venv" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Forbidden data/model artifact: {path.relative_to(ROOT)}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"File exceeds 10 MiB: {path.relative_to(ROOT)}")

    if errors:
        print("Release audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release audit passed: {len(scripts)} clean scripts; no large artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
