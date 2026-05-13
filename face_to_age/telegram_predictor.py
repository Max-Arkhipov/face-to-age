from pathlib import Path
from typing import Optional

import cv2
import lightning as L
import numpy as np
import torch
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from face_to_age.data import build_transforms
from face_to_age.lightning import AgeRegressionModule
from face_to_age.model import AgeModel
from face_to_age.utils import crop_image, get_alignment_transformation


class SingleImageDataset(Dataset):
    def __init__(self, image_bgr: np.ndarray, cfg):
        self.image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self.transform = build_transforms(cfg.preprocessing, train=False)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.transform(self.image), "telegram_input"


class TelegramFacePredictor:
    def __init__(self, cfg, checkpoint_path: str):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cfg = cfg

        # FACE DETECTOR
        self.detector = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.detector.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)

        # PREPROCESS PARAMS
        self.input_size = cfg.preprocessing.image.input_size
        self.input_extension = cfg.preprocessing.image.input_extension
        self.bbox_extension = cfg.preprocessing.image.bbox_extension

        # LOAD MODEL
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.train_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

        base_model = AgeModel(self.train_cfg)

        self.module = AgeRegressionModule.load_from_checkpoint(
            ckpt_path,
            model=base_model,
            cfg=self.train_cfg,
            strict=False,
            weights_only=False,
        )

        self.module.to(self.device)
        self.module.eval()

        self.trainer = L.Trainer(accelerator="auto", logger=False, enable_progress_bar=False)

        print(f"Model loaded successfully on {self.device}")

    def align_and_crop(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None:
            return None

        faces = self.detector.get(img)
        if len(faces) == 0:
            print("Лицо не найдено")
            return None

        face = max(faces, key=lambda x: x.det_score)
        lm = face.landmark_2d_106

        eye_left = np.mean(lm[35:42], axis=0).astype(int)
        eye_right = np.mean(lm[89:96], axis=0).astype(int)
        mouth_avg = np.mean(lm[52:72], axis=0).astype(int)

        aligned_bbox = get_alignment_transformation(
            mouth_avg=mouth_avg,
            eye_left=eye_left,
            eye_right=eye_right,
            eye_to_eye_scale_multipler=1.92,
            eye_to_mouth_scale_multipler=1.89,
        )

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

    @torch.inference_mode()
    def predict(self, img: np.ndarray) -> dict:
        # 1. Детекция и выравнивание
        aligned_img = self.align_and_crop(img)
        if aligned_img is None:
            return {"error": "Лицо не найдено"}

        # 2. DataLoader
        dataset = SingleImageDataset(aligned_img, self.train_cfg)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

        # 3. trainer.predict → predict_step
        predictions = self.trainer.predict(self.module, dataloaders=dataloader)

        result = predictions[0]
        preds = result["preds"]
        uncertainty = result["uncertainty"]
        preds = preds.squeeze()
        age = float(preds.item())

        if uncertainty is not None:
            uncertainty = float(uncertainty.squeeze().item())

        return {
            "predicted_age": round(age, 1),
            "uncertainty": round(uncertainty, 1) if uncertainty is not None else None,
            "aligned_image": aligned_img,
        }
