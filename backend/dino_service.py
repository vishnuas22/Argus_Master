"""Singleton DINOv2 ViT-B/14 embedding service (docs 8.2 module E/F).

Loads once, shared by the realness probe (E) and perturbation probe (F).
CPU-only; images embedded at 224x224 (16x16 patches of 14px) to fit the
<=10s panel budget — documented deviation from the 518px default
(see DECISIONS.md). Lazy imports so the API can boot before torch installs.
"""
import logging
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

logger = logging.getLogger("argus.dino")

MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
IMG_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoService:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._torch = None
        self._lock = threading.Lock()
        self._cache = OrderedDict()  # sha -> embedding
        self._load_error = None

    @classmethod
    def get(cls) -> "DinoService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self) -> bool:
        try:
            import timm  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def ready(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(f"backbone_unavailable: {self._load_error}")
        with self._lock:
            if self._model is not None:
                return
            try:
                import timm
                import torch

                torch.manual_seed(0)
                torch.set_num_threads(8)
                logger.info("loading DINOv2 %s (img_size=%d, CPU)...", MODEL_NAME, IMG_SIZE)
                model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=IMG_SIZE)
                model.eval()
                self._torch = torch
                self._model = model
                logger.info("DINOv2 loaded")
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)[:200]
                logger.error("DINOv2 load failed: %s", exc)
                raise RuntimeError(f"backbone_unavailable: {self._load_error}")

    def warm(self):
        try:
            self._ensure_model()
        except Exception:
            pass

    def _to_tensor(self, pil: Image.Image):
        img = pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return self._torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    def embed(self, pil: Image.Image, cache_key: str = None) -> np.ndarray:
        """L2-normalized 768-d embedding."""
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        self._ensure_model()
        with self._lock:
            with self._torch.no_grad():
                feat = self._model(self._to_tensor(pil)).squeeze(0).numpy().astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        if cache_key is not None:
            self._cache[cache_key] = feat
            while len(self._cache) > 32:
                self._cache.popitem(last=False)
        return feat
