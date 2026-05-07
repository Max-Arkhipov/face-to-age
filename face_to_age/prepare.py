# face_to_age/prepare.py
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def crop_image(
    img: np.ndarray,
    bbox: List[int],
    out_size: Tuple[int],
    margin: Tuple[float] = (0, 0),
    one_based_bbox: bool = True,
):
    """Обрезка по 8-точечному bounding box"""
    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])

    if one_based_bbox:
        A = A - 1
        B = B - 1
        C = C - 1
        D = D - 1

    ext_A = A + (A - B) * margin[0] + (A - D) * margin[1]
    ext_B = B + (B - A) * margin[0] + (B - C) * margin[1]
    ext_C = C + (C - D) * margin[0] + (C - B) * margin[1]

    pts1 = np.float32([ext_A, ext_B, ext_C])
    pts2 = np.float32([[0, 0], [out_size[0] - 1, 0], [out_size[0] - 1, out_size[1] - 1]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (out_size[0], out_size[1]))
    return dst, M


def normalize_image(img, bbox, input_size, input_ext, bbox_ext):
    """Нормализация изображения для обучения"""
    out_size = (
        int(input_size[0] * (1 + 2 * input_ext[0])),
        int(input_size[1] * (1 + 2 * input_ext[1])),
    )
    margin = (
        input_ext[0] + bbox_ext[0] + 2 * input_ext[0] * bbox_ext[0],
        input_ext[1] + bbox_ext[1] + 2 * input_ext[1] * bbox_ext[1],
    )
    out_img, _ = crop_image(img, bbox, out_size, margin, one_based_bbox=True)
    return out_img


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def prepare(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print("=" * 80)
    print("CONFIG:")
    print(cfg)
    print("=" * 80)

    print("=== Подготовка датасета ===")

    base_out_dir = Path(cfg.paths.processed_dir) / cfg.dataset.name
    print(f"Папка для сохранения: {base_out_dir}\n")

    # Создаём папки train/val/test
    for split in ["train", "val", "test"]:
        (base_out_dir / split).mkdir(parents=True, exist_ok=True)

    # Загружаем benchmark
    benchmark_path = Path(cfg.dataset.benchmark)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmarks = OmegaConf.load(f)

    total_processed = 0

    for benchmark in benchmarks:
        print(f"Обработка: {benchmark.database}")

        db_path = Path(benchmark.database)
        if not db_path.exists():
            print(f"Не найден: {benchmark.database}")
            continue

        with open(db_path, "r", encoding="utf-8") as f:
            db = OmegaConf.load(f)

        split_map = {}
        for split_def in benchmark.split:
            for part_idx, part in enumerate(["trn", "val", "tst"]):
                split_name = ["train", "val", "test"][part_idx]
                for folder_id in split_def.get(part, []):
                    split_map[folder_id] = split_name

        processed = skipped = 0

        for face in tqdm(db, desc=cfg.dataset.name, leave=False):
            folder = face.get("folder")
            if folder not in split_map:
                skipped += 1
                continue

            img_path = Path(cfg.paths.data_dir) / face["img_path"]
            if not img_path.exists():
                skipped += 1
                continue

            aligned_bbox = face.get("aligned_bbox")
            if not aligned_bbox or len(aligned_bbox) != 8:
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            try:
                cropped = normalize_image(
                    img=img,
                    bbox=aligned_bbox,
                    input_size=cfg.preprocessing.image.input_size,
                    input_ext=cfg.preprocessing.image.input_extension,
                    bbox_ext=cfg.preprocessing.image.bbox_extension,
                )

                save_path = base_out_dir / split_map[folder] / img_path.name
                cv2.imwrite(str(save_path), cropped)

                processed += 1
                total_processed += 1
            except Exception:
                skipped += 1
                continue

        print(f"Принято: {processed} | Пропущено: {skipped}")

    print(f"\nВсего обработано: {total_processed} изображений")
    print(f"Данные добавлены в: {base_out_dir}")


if __name__ == "__main__":
    prepare()
