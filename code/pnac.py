#!/usr/bin/env python3
"""PNAC: Pseudolabel Amplification Cascade for label-noise quantification.

This script trains a baseline model on a labeled split, then iteratively
adds high-confidence pseudo-labels from an unlabeled pool and tracks F1
degradation on a clean validation split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover - for environments without torch
    raise SystemExit(
        "PyTorch/torchvision not available. Install requirements first."
    ) from exc


@dataclass
class DatasetConfig:
    data_root: Path
    manifest_path: Path
    image_size: int = 224
    normalization_mean: Tuple[float, float, float] = (0.131416803, 0.136298867, 0.129060801)
    normalization_std: Tuple[float, float, float] = (0.165112494, 0.167405856, 0.163927364)


class ManifestDataset(Dataset):
    def __init__(self, root: Path, items: List[Tuple[str, int]], transform=None):
        self.root = root
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path, label = self.items[idx]
        img_path = self.root / rel_path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class UnlabeledDataset(Dataset):
    def __init__(self, root: Path, files: List[str]):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel_path = self.files[idx]
        img_path = self.root / rel_path
        image = Image.open(img_path).convert("RGB")
        return image, rel_path


class TTADataset(Dataset):
    """Dataset that loads images and applies TTA transforms in parallel workers."""
    def __init__(self, root: Path, files: List[str], tta_transform, tta_runs: int):
        self.root = root
        self.files = files
        self.tta_transform = tta_transform
        self.tta_runs = tta_runs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel_path = self.files[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        tensors = torch.stack([self.tta_transform(img) for _ in range(self.tta_runs)])
        return tensors, rel_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def load_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    splits = data["splits"]
    class_names = sorted(splits["train"].keys())
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    def build_split(split_name: str) -> List[Tuple[str, int]]:
        items = []
        for cls, files in splits[split_name].items():
            label = label_to_idx[cls]
            items.extend([(fp, label) for fp in files])
        return items

    train_items = build_split("train")
    val_items = build_split("validation")
    test_items = build_split("test")
    unlabeled_files = data.get("unlabeled_pool", [])
    return class_names, label_to_idx, train_items, val_items, test_items, unlabeled_files


def inject_uniform_noise(items: List[Tuple[str, int]], num_classes: int, noise_rate: float, seed: int):
    if noise_rate <= 0:
        return items
    rng = random.Random(seed)
    noisy = []
    for path, label in items:
        if rng.random() < noise_rate:
            candidates = [c for c in range(num_classes) if c != label]
            label = rng.choice(candidates)
        noisy.append((path, label))
    return noisy


def build_transforms(cfg: DatasetConfig):
    train_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalization_mean, cfg.normalization_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalization_mean, cfg.normalization_std),
        ]
    )
    tta_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalization_mean, cfg.normalization_std),
        ]
    )
    return train_tf, eval_tf, tta_tf


def build_model(num_classes: int, pretrained: bool = False):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device, scaler=None, desc="Training"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    use_amp = scaler is not None
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(loader.dataset))


def evaluate(model, loader, device, num_classes: int) -> Tuple[float, float, List[float]]:
    """Evaluate model on a data loader.

    Returns:
        Tuple of (accuracy, macro_f1, per_class_f1_list).
    """
    model.eval()
    all_preds = []
    all_labels = []
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    accuracy = sum(p == y for p, y in zip(all_preds, all_labels)) / max(1, len(all_labels))
    macro_f1_score, per_class_f1 = macro_f1(all_labels, all_preds, num_classes)
    return accuracy, macro_f1_score, per_class_f1


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> Tuple[float, List[float]]:
    """Calculate macro F1 score and per-class F1 scores.

    Returns:
        Tuple of (macro_mean, per_class_list) where macro_mean is the average F1
        and per_class_list contains individual F1 scores for each class.
    """
    f1s = []
    for cls in range(num_classes):
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    macro_mean = float(sum(f1s) / max(1, len(f1s)))
    return macro_mean, f1s


def tta_predict(
    model,
    files: List[str],
    root: Path,
    tta_transform,
    device,
    num_classes: int,
    tta_runs: int,
    batch_size: int,
    num_workers: int = 2,
):
    model.eval()
    results = {}
    use_amp = device.type == 'cuda'
    use_cuda = device.type == 'cuda'
    # Use DataLoader with parallel workers for I/O
    tta_batch_size = max(8, batch_size // tta_runs)
    dataset = TTADataset(root, files, tta_transform, tta_runs)
    loader = DataLoader(
        dataset, batch_size=tta_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )
    with torch.no_grad():
        for tensors, paths in tqdm(loader, desc="Pseudo-labeling", leave=False):
            # tensors shape: (B, tta_runs, C, H, W)
            B, K, C, H, W = tensors.shape
            tensors = tensors.view(B * K, C, H, W).to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(tensors)
            else:
                logits = model(tensors)
            probs = torch.softmax(logits, dim=1)
            probs = probs.view(B, K, num_classes).mean(dim=1)
            probs = probs.cpu().numpy()
            for path, prob in zip(paths, probs):
                results[path] = prob
    return results


def fit_decay_rate(f1_scores: List[float]) -> float:
    if len(f1_scores) < 2:
        return 0.0
    baseline = f1_scores[0]
    deltas = [max(0.0, baseline - f1) for f1 in f1_scores]
    alpha = max(deltas) if max(deltas) > 0 else 1e-6
    xs = []
    ys = []
    for t, delta in enumerate(deltas, start=1):
        ratio = 1.0 - min(delta / alpha, 0.999999)
        if ratio <= 0:
            continue
        xs.append(float(t))
        ys.append(math.log(ratio))
    if len(xs) < 2:
        return 0.0
    slope = np.polyfit(xs, ys, deg=1)[0]
    beta = -slope
    return float(beta)


def save_metrics(path: Path, rows: List[dict]):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="PNAC experiment runner")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--manifest", type=Path, required=True, help="Split manifest JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Output folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (manuscript uses 160 on RTX 3090)")
    parser.add_argument("--tta", type=int, default=10, help="TTA runs per image")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--noise-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--max-unlabeled", type=int, default=0)
    parser.add_argument("--max-labeled", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Windows shared memory fix: limit workers to avoid error 1455
    import sys
    if sys.platform == 'win32' and args.num_workers > 2:
        print(f"Note: Reducing num_workers from {args.num_workers} to 2 for Windows compatibility")
        args.num_workers = 2

    set_seed(args.seed)

    cfg = DatasetConfig(data_root=args.data_root, manifest_path=args.manifest)
    class_names, _, train_items, val_items, test_items, unlabeled_files = load_manifest(cfg.manifest_path)

    if args.max_labeled > 0:
        train_items = train_items[: args.max_labeled]
    if args.max_unlabeled > 0:
        unlabeled_files = unlabeled_files[: args.max_unlabeled]

    train_items = inject_uniform_noise(train_items, len(class_names), args.noise_rate, args.seed)

    train_tf, eval_tf, tta_tf = build_transforms(cfg)
    val_dataset = ManifestDataset(cfg.data_root, val_items, transform=eval_tf)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None
    )

    device = torch.device(args.device)

    # Setup mixed-precision training (AMP) for faster GPU computation
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed-precision training (AMP) enabled for faster GPU computation")

    # Print configuration
    print(f"\n{'='*60}")
    print(f"PNAC Experiment - Noise Rate: {args.noise_rate}")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Classes: {class_names}")
    print(f"  Train samples: {len(train_items)}")
    print(f"  Validation samples: {len(val_items)}")
    print(f"  Unlabeled pool: {len(unlabeled_files)}")
    print(f"  Epochs per iteration: {args.epochs}")
    print(f"  Total iterations: {args.iterations}")
    print(f"{'='*60}")

    # Use output directory directly if provided, else create timestamped subdirectory
    if args.output_dir.name.startswith("noise_"):
        # Called from run_calibration.py with specific noise folder
        run_dir = args.output_dir
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = args.output_dir / f"pnac_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    remaining_unlabeled = list(unlabeled_files)
    pseudo_accumulated: List[Tuple[str, int]] = []

    metrics = []
    f1_scores = []
    per_class_f1_history: List[List[float]] = []  # Track per-class F1 at each iteration

    for iteration in range(args.iterations + 1):
        print(f"\n[Iteration {iteration}/{args.iterations}]")

        model = build_model(len(class_names), pretrained=args.pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        current_items = list(train_items)
        current_items.extend(pseudo_accumulated)
        pseudo_path = run_dir / "pseudo_labels" / f"iter_{iteration:02d}.json"
        pseudo_path.parent.mkdir(parents=True, exist_ok=True)

        train_dataset = ManifestDataset(cfg.data_root, current_items, transform=train_tf)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None
        )

        print(f"  Training on {len(current_items)} samples ({len(pseudo_accumulated)} pseudo-labels)...")
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device, scaler, desc=f"Epoch {epoch+1}/{args.epochs}")
            print(f"    Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

        print(f"  Evaluating...")
        acc, f1, per_class_f1 = evaluate(model, val_loader, device, len(class_names))
        print(f"  -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")

        # Save model checkpoint
        model_path = run_dir / "models" / f"model_iter_{iteration:02d}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': acc,
            'f1': f1,
            'noise_rate': args.noise_rate,
        }, model_path)
        print(f"  -> Model saved: {model_path.name}")
        f1_scores.append(f1)
        per_class_f1_history.append(per_class_f1)

        # Build metrics row with per-class F1 columns
        row = {
            "iteration": iteration,
            "val_accuracy": acc,
            "val_f1": f1,
            "labeled_size": len(current_items),
            "unlabeled_remaining": len(remaining_unlabeled),
        }
        # Add per-class F1 scores dynamically based on number of classes
        for i, class_f1 in enumerate(per_class_f1):
            row[f"f1_class_{i}"] = class_f1
        metrics.append(row)

        if iteration == args.iterations:
            break

        # Pseudo-label step for next iteration
        print(f"  Pseudo-labeling {len(remaining_unlabeled)} unlabeled samples (TTA={args.tta})...")
        all_probs = tta_predict(
            model,
            remaining_unlabeled,
            cfg.data_root,
            tta_tf,
            device,
            len(class_names),
            args.tta,
            args.batch_size,
            num_workers=args.num_workers,
        )

        selected = []
        new_remaining = []
        for rel_path in remaining_unlabeled:
            prob = all_probs.get(rel_path)
            if prob is None:
                continue
            label = int(np.argmax(prob))
            conf = float(np.max(prob))
            if conf >= args.confidence:
                selected.append({"path": rel_path, "label": label, "confidence": conf})
            else:
                new_remaining.append(rel_path)

        pseudo_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        pseudo_accumulated.extend([(item["path"], item["label"]) for item in selected])
        remaining_unlabeled = new_remaining
        print(f"  -> Selected {len(selected)} new pseudo-labels (conf >= {args.confidence})")

    decay_beta = fit_decay_rate(f1_scores)
    print(f"\n{'='*60}")
    print(f"COMPLETED - Noise Rate: {args.noise_rate}")
    print(f"{'='*60}")
    print(f"  F1 scores: {[f'{f:.4f}' for f in f1_scores]}")
    print(f"  Decay rate (beta): {decay_beta:.4f}")
    print(f"{'='*60}\n")
    summary = {
        "class_names": class_names,
        "noise_rate": args.noise_rate,
        "confidence": args.confidence,
        "tta": args.tta,
        "iterations": args.iterations,
        "decay_beta": decay_beta,
        "f1_scores": f1_scores,
        "per_class_f1_scores": per_class_f1_history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metrics(run_dir / "metrics.csv", metrics)


if __name__ == "__main__":
    main()
