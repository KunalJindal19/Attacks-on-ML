"""
train_victim.py
===============
Train a DenseNet-121 victim model on the MEMBER-ONLY portion of the NIH
Chest X-ray dataset.

Design choices to encourage intentional overfitting (for MIA signal):
  ✓ Full fine-tune: all layers trainable
  ✓ Pretrained on ImageNet (pretrained=True)
  ✓ NO dropout, NO weight decay, NO data augmentation
  ✓ Loss: BCEWithLogitsLoss (multi-label)
  ✓ Optimizer: Adam, lr=1e-4
  ✓ 30 epochs → expect train_acc > 99%, val_acc ~93%, gap ~6%

Inputs
------
  Victim_Model/manifest.csv   (produced by prepare_dataset.py)

Outputs
-------
  Victim_Model/victim.pth          ← model weights (state_dict)
  Victim_Model/victim_meta.json    ← architecture info + final metrics

Usage
-----
  python train_victim.py [--epochs N] [--batch_size B] [--lr LR] [--arch ARCH]
"""

import os
import sys
import ast
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "victim.pth")
META_PATH     = os.path.join(BASE_DIR, "victim_meta.json")

DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema",   "Fibrosis",       "Effusion",     "Pneumonia",    "Pleural_Thickening",
    "Cardiomegaly","Nodule",          "Mass",         "Hernia",       "No Finding",
]
NUM_CLASSES = len(DISEASE_CLASSES)  # 15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Defaults (can be overridden via CLI)
NUM_EPOCHS    = 30
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
IMG_SIZE      = 224
VAL_FRACTION  = 0.15     # fraction of members held out for monitoring
RANDOM_SEED   = 42


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NIHDataset(Dataset):
    """PyTorch Dataset for NIH Chest X-rays with multi-label targets."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"  Warning: failed to load {row['path']}: {e}", flush=True)
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        # label_idx stored as string "[1,0,0,...]" in manifest.csv
        label_list = ast.literal_eval(row["label_idx"])
        return img, torch.tensor(label_list, dtype=torch.float32)


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(architecture: str, num_classes: int) -> nn.Module:
    """Build a pretrained model with a custom multi-label head.

    Currently supported: densenet121 (default), resnet50, efficientnet_b3.
    Head: Linear(in_features, 512) → ReLU → Linear(512, num_classes)
    No dropout, no weight_decay — intentional for overfitting.
    """
    arch = architecture.lower()

    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f  = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f  = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            "Choose from: densenet121, resnet50, efficientnet_b3"
        )

    # Full fine-tune — all layers trainable
    for p in model.parameters():
        p.requires_grad = True

    return model


# ─── Eval ─────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, num_classes) -> tuple[float, float]:
    """Compute average loss and element-wise accuracy on a DataLoader."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_n       += images.size(0) * num_classes

    avg_loss = total_loss / (total_n / num_classes)
    accuracy = total_correct / total_n
    return avg_loss, accuracy


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NIH victim model")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--arch", type=str, default="densenet121",
        help="Model architecture: densenet121 | resnet50 | efficientnet_b3"
    )
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device:       {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU:          {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Architecture: {args.arch}")
    sys.stdout.flush()

    # ── Load manifest, keep only members ──────────────────────────────────────
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run prepare_dataset.py first."
        )
    manifest  = pd.read_csv(MANIFEST_PATH)
    member_df = manifest[manifest["split"] == "member"].reset_index(drop=True)

    # Infer num_classes from stored label vector
    sample_label = ast.literal_eval(member_df["label_idx"].iloc[0])
    num_classes  = len(sample_label)
    label_names  = DISEASE_CLASSES[:num_classes]

    print(f"[INFO] Member images: {len(member_df)}")
    print(f"[INFO] Num classes:   {num_classes}")
    print(f"[INFO] Class names:   {label_names}")
    sys.stdout.flush()

    # ── Train / val split (within members) ────────────────────────────────────
    n_val   = int(len(member_df) * VAL_FRACTION)
    shuffled = member_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df   = shuffled.iloc[:n_val].reset_index(drop=True)
    train_df = shuffled.iloc[n_val:].reset_index(drop=True)

    print(f"[INFO] Training on {len(train_df)} images, validating on {len(val_df)}.")
    sys.stdout.flush()

    pin = device.type == "cuda"
    train_ds     = NIHDataset(train_df)
    val_ds       = NIHDataset(val_df)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=pin,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=pin,
    )

    # ── Model, optimizer, criterion ───────────────────────────────────────────
    model     = build_model(args.arch, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[TRAIN] Starting training …", flush=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss    += loss.item() * images.size(0)
            preds          = (torch.sigmoid(logits) > 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_n       += images.size(0) * num_classes

        train_loss = epoch_loss / (epoch_n / num_classes)
        train_acc  = epoch_correct / epoch_n
        val_loss, val_acc = evaluate(model, val_loader, device, num_classes)
        gap_pct    = (train_acc - val_acc) * 100
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"gap={gap_pct:+.2f}%  ({elapsed:.1f}s)",
            flush=True,
        )
        sys.stdout.flush()

    # ── Save model + metadata ─────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)

    final_train_acc = history["train_acc"][-1]
    final_val_acc   = history["val_acc"][-1]
    memorization_gap = final_train_acc - final_val_acc

    meta = {
        "architecture":    args.arch,
        "num_classes":     num_classes,
        "label_names":     label_names,
        "img_size":        IMG_SIZE,
        "imagenet_mean":   IMAGENET_MEAN,
        "imagenet_std":    IMAGENET_STD,
        "final_train_acc": final_train_acc,
        "final_val_acc":   final_val_acc,
        "memorization_gap": memorization_gap,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "learning_rate":   args.lr,
        "weight_decay":    0.0,
        "dropout":         False,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[DONE]")
    print(f"  Model:            {MODEL_PATH}")
    print(f"  Metadata:         {META_PATH}")
    print(f"  Final train_acc:  {final_train_acc:.4f}")
    print(f"  Final val_acc:    {final_val_acc:.4f}")
    print(f"  Memorization gap: {memorization_gap * 100:+.2f}%")
    sys.stdout.flush()

    if memorization_gap < 0.05:
        print(
            "\n  WARNING: Memorization gap is <5%. The MIA signal may be weak. "
            "Consider increasing --epochs or checking that no regularization is applied.",
            flush=True,
        )


if __name__ == "__main__":
    main()
