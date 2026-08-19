"""Fail when release notebooks or files contain common publication hazards."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SOURCE = {
    "pinned CUDA device": re.compile(r"CUDA_VISIBLE_DEVICES|cuda:\d+"),
    "GPU shell probe": re.compile(r"nvidia-smi", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\"),
    "Colab drive path": re.compile(r"/content/drive/"),
}
FORBIDDEN_SUFFIXES = {".mat", ".pt", ".pth", ".ckpt", ".pkl", ".joblib", ".onnx"}


def audit_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if "widgets" in notebook.get("metadata", {}):
        errors.append(f"{path}: widget state is present")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"{path}: cell {index} has an execution count")
        if cell.get("outputs"):
            errors.append(f"{path}: cell {index} has output")
        source = "".join(cell.get("source", []))
        for label, pattern in FORBIDDEN_SOURCE.items():
            if pattern.search(source):
                errors.append(f"{path}: cell {index} contains {label}")
        try:
            compile(source, f"{path}:cell-{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"{path}: cell {index} is not valid Python: {exc.msg}")
    return errors


def main() -> int:
    errors: list[str] = []
    notebooks = sorted(
        path for path in ROOT.rglob("*.ipynb") if ".venv" not in path.parts
    )
    if not notebooks:
        errors.append("No curated notebooks found")
    for path in notebooks:
        errors.extend(audit_notebook(path))

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
    print(f"Release audit passed: {len(notebooks)} clean notebooks; no large artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
