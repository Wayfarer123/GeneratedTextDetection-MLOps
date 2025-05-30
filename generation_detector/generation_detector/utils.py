import warnings
from pathlib import Path

from dvc.repo import Repo


def pull_dvc_data(*targets: str, dvc_root_path: Path | None = None) -> None:
    """
    Pulls DVC tracked data using DVC Python API.

    Args:
        targets: Tuple of DVC target files/directories to pull (e.g., "data/raw.dvc").
                 If empty, pulls all tracked data.
        dvc_root_path: Path to the DVC project root. If None, assumes current dir.
    """
    root_path = str(dvc_root_path) if dvc_root_path else "."

    with Repo(root_path) as repo:
        if not targets:
            repo.pull()
        else:
            repo.pull(targets=list(targets))
        print("DVC pull successful.")


def get_project_root() -> Path:
    """Returns the project root directory by searching for .git or pyproject.toml."""
    current_path = Path.cwd()
    while True:
        if (current_path / ".git").exists() or (
            current_path / "pyproject.toml"
        ).exists():
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:  # Reached filesystem root
            # Fallback to current working directory if no markers found
            warnings.warn(
                "Could not find project root markers (.git, pyproject.toml). Assuming CWD is root.",
                UserWarning,
                stacklevel=2,
            )
            return Path.cwd()
        current_path = parent_path
