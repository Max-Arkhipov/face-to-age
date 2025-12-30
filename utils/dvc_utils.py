import subprocess
from pathlib import Path


def dvc_pull_if_needed(paths, remote="gdrive"):
    for path in paths:
        path = Path(path)
        if not path.exists() or not any(path.iterdir()):
            print(f"{path} пуст или отсутствует, выполняем dvc pull...")
            subprocess.run(["dvc", "pull", str(path), "-r", remote], check=True)
        else:
            print(f"{path} существует, пропускаем dvc pull.")
