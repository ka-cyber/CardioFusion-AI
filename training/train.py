"""
Training Entry Point
=======================

Config-driven training loop for any of the fusion strategies in
`models/fusion/fusion_models.py`. Run with:

    python -m training.train --config configs/default.yaml

This script assumes `ECGPPGWindowDataset` (see `training/dataset.py`) has
already been built from preprocessed, cached signal windows -- it does not
re-run filtering/synchronization itself (that's an offline step; see
`preprocessing/`).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.cnn.cnn_models import CNN1D
from models.fusion.fusion_models import (
    EarlyFusion, FeatureLevelFusion, AttentionFusion, LateDecisionFusion, AdaptiveDynamicFusion,
)
from training.dataset import ECGPPGWindowDataset
from evaluation.evaluate import compute_classification_metrics
from utils.config import load_config


FUSION_REGISTRY = {
    "early": EarlyFusion,
    "feature_level": FeatureLevelFusion,
    "attention": AttentionFusion,
    "late_decision": LateDecisionFusion,
    "adaptive_dynamic": AdaptiveDynamicFusion,
}


def build_model(cfg: dict) -> nn.Module:
    fusion_type = cfg["model"]["fusion_type"]
    embedding_dim = cfg["model"].get("embedding_dim", 128)
    num_classes = cfg["model"].get("num_classes", 2)

    if fusion_type == "early":
        return EarlyFusion(num_classes=num_classes, embedding_dim=embedding_dim)

    # Fusion strategies that operate on independent per-modality encoders
    ecg_encoder = CNN1D(in_channels=1, num_classes=num_classes, embedding_dim=embedding_dim)
    ppg_encoder = CNN1D(in_channels=1, num_classes=num_classes, embedding_dim=embedding_dim)

    if fusion_type == "feature_level":
        return FeatureLevelFusion(ecg_encoder, ppg_encoder, embedding_dim, num_classes)
    if fusion_type == "attention":
        return AttentionFusion(ecg_encoder, ppg_encoder, embedding_dim, num_classes)
    if fusion_type == "late_decision":
        return LateDecisionFusion(ecg_encoder, ppg_encoder, num_classes)
    if fusion_type == "adaptive_dynamic":
        return AdaptiveDynamicFusion(ecg_encoder, ppg_encoder, embedding_dim, num_classes)

    raise ValueError(f"Unknown fusion_type '{fusion_type}'. Choose from {list(FUSION_REGISTRY)}")


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for ecg, ppg, labels in loader:
            ecg, ppg, labels = ecg.to(device), ppg.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(ecg, ppg)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    metrics = compute_classification_metrics(labels_np, preds)
    metrics["loss"] = avg_loss
    return metrics


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # NOTE: Replace this with real cached ECG/PPG window arrays -- see
    # training/dataset.py and preprocessing/ for how to produce them from
    # raw PhysioNet/PulseDB/BIDMC recordings. This entry point intentionally
    # does not fabricate or assume any specific dataset content.
    dataset_cfg = cfg["data"]
    ecg_signals = np.load(dataset_cfg["ecg_signals_path"], allow_pickle=True)
    ppg_signals = np.load(dataset_cfg["ppg_signals_path"], allow_pickle=True)
    labels = np.load(dataset_cfg["labels_path"], allow_pickle=True)

    full_dataset = ECGPPGWindowDataset(
        list(ecg_signals), list(ppg_signals), list(labels),
        window_size=dataset_cfg.get("window_size", 1000),
        stride=dataset_cfg.get("stride", 500),
    )

    val_frac = cfg["training"].get("val_fraction", 0.2)
    n_val = int(len(full_dataset) * val_frac)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("lr", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1.0, None
    checkpoint_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} val_auroc={val_metrics.get('auroc', float('nan')):.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_dir / "best_model.pt")

    print(f"[train] done. best val_f1={best_val_f1:.4f}, checkpoint saved to {checkpoint_dir / 'best_model.pt'}")
    return model, best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CardioFusion-AI fusion model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
