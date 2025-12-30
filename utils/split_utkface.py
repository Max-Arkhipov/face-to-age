import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DIR = Path("data/UTKFace")
OUT_DIR = Path("data")

PREDICT_RATIO = 0.05

# =========================
random.seed(SEED)


def is_valid_utkface_image(path: Path) -> bool:
    """
    UTKFace filename format:
    age_gender_race_date.jpg
    """
    parts = path.stem.split("_")
    if len(parts) < 4:
        return False
    return parts[0].isdigit()


def main():
    images = [p for p in RAW_DIR.glob("*.jpg") if is_valid_utkface_image(p)]

    print(f"Found {len(images)} valid UTKFace images")

    random.shuffle(images)

    # -------------------------
    # Predict split (optional)
    # -------------------------
    n_predict = int(len(images) * PREDICT_RATIO)
    predict_images = images[:n_predict]
    remaining = images[n_predict:]

    # -------------------------
    # Train / Val / Test
    # -------------------------
    n_total = len(remaining)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": remaining[:n_train],
        "val": remaining[n_train : n_train + n_val],
        "test": remaining[n_train + n_val :],
    }

    # -------------------------
    # Copy files
    # -------------------------
    for split, files in splits.items():
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for img in files:
            shutil.copy(img, split_dir / img.name)

    # -------------------------
    # Predict (no labels)
    # -------------------------
    predict_dir = OUT_DIR / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)

    for img in predict_images:
        shutil.copy(img, predict_dir / img.name)

    # -------------------------
    # Summary
    # -------------------------
    print("  Dataset split completed:")
    print(f"  Train:   {len(splits['train'])}")
    print(f"  Val:     {len(splits['val'])}")
    print(f"  Test:    {len(splits['test'])}")
    print(f"  Predict: {len(predict_images)}")


if __name__ == "__main__":
    main()
