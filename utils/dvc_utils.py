import subprocess
from pathlib import Path


def ensure_data(path: Path):
    """
    Ensures that data is available locally via DVC.
    """
    if path.exists() and any(path.iterdir()):
        return

    subprocess.run(
        ["dvc", "pull", str(path)],
        check=True,
    )
