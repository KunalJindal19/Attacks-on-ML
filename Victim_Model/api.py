"""
api.py
======
VictimAPI — a black-box wrapper around the trained victim model.

Simulates the API that an attacker would query: given image file paths,
return (N, 15) sigmoid confidence scores.

The architecture is read from victim_meta.json so the same api.py works
regardless of which --arch was used in train_victim.py.

Usage
-----
  from api import VictimAPI
  api = VictimAPI("victim.pth")                     # loads meta from victim_meta.json
  scores = api.predict(np.array(image_paths))       # shape (N, 15)
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class VictimAPI:
    """Black-box API wrapper for the trained NIH victim model.

    Parameters
    ----------
    model_path : str
        Path to ``victim.pth`` (state_dict).  The companion
        ``victim_meta.json`` must exist in the same directory.
    num_classes : int, optional
        Number of output classes.  Read from metadata if not provided.
    batch_size : int
        Inference batch size.
    device : str, optional
        'cuda' or 'cpu'.  Auto-detected when None.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_path: str,
        num_classes: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.batch_size = batch_size

        # ── Load metadata ──────────────────────────────────────────────────────
        meta_path = model_path.replace(".pth", "_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}. "
                "Run train_victim.py first."
            )
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if num_classes is None:
            num_classes = int(meta["num_classes"])

        self.num_classes  = num_classes
        self.architecture = meta.get("architecture", "densenet121")
        self.img_size     = int(meta.get("img_size", 224))
        self.label_names  = meta.get("label_names", [])

        # ── Device ─────────────────────────────────────────────────────────────
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Build architecture and load weights ────────────────────────────────
        self.model = self._build_model()
        state_dict = torch.load(
            model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # ── Preprocessing (must match training) ────────────────────────────────
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    def _build_model(self) -> nn.Module:
        """Reconstruct model architecture (must match train_victim.py)."""
        arch = self.architecture.lower()
        nc   = self.num_classes

        if arch == "densenet121":
            model = models.densenet121(weights=None)
            in_f  = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 512),
                nn.ReLU(),
                nn.Linear(512, nc),
            )

        elif arch == "resnet50":
            model = models.resnet50(weights=None)
            in_f  = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_f, 512),
                nn.ReLU(),
                nn.Linear(512, nc),
            )

        elif arch == "efficientnet_b3":
            model = models.efficientnet_b3(weights=None)
            in_f  = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 512),
                nn.ReLU(),
                nn.Linear(512, nc),
            )

        else:
            raise ValueError(
                f"Unknown architecture in metadata: '{self.architecture}'. "
                "Supported: densenet121, resnet50, efficientnet_b3"
            )

        return model

    # ------------------------------------------------------------------
    def _load_image(self, file_path: str) -> torch.Tensor:
        """Load and preprocess a single image → (3, H, W) tensor."""
        try:
            img = Image.open(file_path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"  Warning: failed to load {file_path}: {e}", flush=True)
            return torch.zeros(3, self.img_size, self.img_size)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid confidence scores for an array of image paths.

        Parameters
        ----------
        X : np.ndarray of str, shape (N,)
            Absolute paths to image files.

        Returns
        -------
        probs : np.ndarray of float32, shape (N, num_classes)
            Sigmoid confidence scores in [0, 1].
        """
        n         = len(X)
        all_probs = []

        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                end         = min(start + self.batch_size, n)
                batch_paths = X[start:end]

                tensors = [self._load_image(p) for p in batch_paths]
                batch   = torch.stack(tensors).to(self.device, non_blocking=True)

                logits = self.model(batch)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"VictimAPI(arch={self.architecture}, "
            f"num_classes={self.num_classes}, "
            f"device={self.device}, "
            f"batch_size={self.batch_size})"
        )
