"""Portable project and artifact paths."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing this project's ``pyproject.toml``."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        has_project_file = (candidate / "pyproject.toml").is_file()
        has_package = (candidate / "src" / "battery_soh").is_dir()
        if has_project_file and has_package:
            return candidate
    raise FileNotFoundError("Could not locate the battery-health-forecasting project root")


def artifact_path(name: str | Path, *, create_parent: bool = True) -> Path:
    """Return a safe path below the Git-ignored ``artifacts`` directory."""

    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact name must be a relative path inside artifacts/")
    path = find_project_root() / "artifacts" / relative
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
