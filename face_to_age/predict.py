# face_to_age/predict.py
from pathlib import Path
from typing import Optional

import cv2
import hydra
import numpy as np
from insightface.app import FaceAnalysis
from omegaconf import DictConfig

from face_to_age.utils import crop_image, get_alignment_transformation


class FacePredictor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # === 1. Загружаем Face Detector (RetinaFace) ===
        self.detector = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.detector.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)

        # === 2. Параметры обрезки (как при обучении) ===
        self.input_size = cfg.preprocessing.image.input_size
        self.input_extension = cfg.preprocessing.image.input_extension
        self.bbox_extension = cfg.preprocessing.image.bbox_extension

    def align_and_crop(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Выравнивает и обрезает лицо для подачи в модель"""
        if img is None:
            return None

        # Детекция лица
        faces = self.detector.get(img)
        if len(faces) == 0:
            print("Лицо не найдено")
            return None

        # Берём лицо с наибольшей уверенностью / размером
        face = max(faces, key=lambda x: x.det_score)

        # Извлекаем ключевые точки (InsightFace даёт 106 или 68 точек)
        lm = face.landmark_2d_106

        # Находим координаты по средним
        eye_left = np.mean(lm[35:42], axis=0).astype(int)
        eye_right = np.mean(lm[89:96], axis=0).astype(int)
        mouth_avg = np.mean(lm[52:72], axis=0).astype(int)

        # Получаем выровненный bbox
        aligned_bbox = get_alignment_transformation(
            mouth_avg=mouth_avg,
            eye_left=eye_left,
            eye_right=eye_right,
            eye_to_eye_scale_multipler=1.92,
            eye_to_mouth_scale_multipler=1.89,
        )

        # Обрезаем изображение
        out_size = (
            int(self.input_size[0] * (1 + 2 * self.input_extension[0])),
            int(self.input_size[1] * (1 + 2 * self.input_extension[1])),
        )
        margin = (
            self.input_extension[0]
            + self.bbox_extension[0]
            + 2 * self.input_extension[0] * self.bbox_extension[0],
            self.input_extension[1]
            + self.bbox_extension[1]
            + 2 * self.input_extension[1] * self.bbox_extension[1],
        )

        aligned_img, _ = crop_image(
            img=img,
            bbox=[int(x) for x in aligned_bbox],
            out_size=out_size,
            margin=margin,
            one_based_bbox=True,
        )

        return aligned_img

    def predict(self, image_path: str) -> dict:
        """Полный пайплайн: выравнивание + предсказание"""
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Не удалось прочитать изображение"}

        aligned_img = self.align_and_crop(img)
        if aligned_img is None:
            return {"error": "Лицо не найдено"}

        # === Вызов модели ===
        # 1. Создаем путь к папке (parents=True создаст data, если её нет)
        output_dir = Path("data/predict")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Берем только имя файла из исходного пути
        file_name = Path(image_path).stem + ".jpg"
        output_path = output_dir / file_name
        cv2.imwrite(str(output_path), aligned_img)

        print(f"Изображение выровнено и сохранено как: {output_path}")
        print(f"Размер после обработки: {aligned_img.shape}")

        return {
            "status": "success",
            "aligned_image": str(output_path),
            "original_shape": img.shape,
            "aligned_shape": aligned_img.shape,
        }


# ====================== CLI ======================
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def predict(cfg: DictConfig):
    predictor = FacePredictor(cfg)

    # Папка с изображениями
    image_dir = Path(cfg.get("image_dir", "data/image"))

    if not image_dir.exists():
        print(f"Папка не найдена: {image_dir}")
        return

    # Поддерживаемые форматы
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

    if len(image_paths) == 0:
        print(f"В папке нет изображений: {image_dir}")
        return

    print(f"Найдено изображений: {len(image_paths)}")

    results = []

    for image_path in image_paths:
        print("=" * 80)
        print(f"Processing: {image_path.name}")

        try:
            result = predictor.predict(str(image_path))

            results.append({"file": image_path.name, **result})

            print(result)

        except Exception as e:
            print(f"Ошибка при обработке {image_path.name}: {e}")

    print("=" * 80)
    print("DONE")

    # Финальный summary
    for r in results:
        if "predicted_age" in r:
            print(f"{r['file']} -> age={r['predicted_age']}")


if __name__ == "__main__":
    predict()
