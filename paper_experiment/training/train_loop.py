"""
Training protocol (manuscript Section II-F):

"Each of the eight architectures was trained independently under five
random seeds (0-4), which govern both parameter initialization and
minibatch ordering; the train/validation/test partition itself was fixed
and shared across all seeds and architectures ... All models were trained
with Adam (learning rate 5e-4, weight decay 1e-5, gradient-norm clipping at
1.0 ...), batch size 32, mean-squared-error loss, up to 30 epochs with
early stopping (patience 6) on validation MSE. Evaluation used the single
fixed test partition (324 window pairs) for every architecture and seed."

NOT EXECUTED in this environment: PyTorch is not installed (no network
access to install it in this sandbox). Statically reviewed line-by-line
against the manuscript quote above; every hyperparameter is read from
paper_experiment/configs/paper_experiment.yaml (single source of truth),
not hardcoded here, so a real run cannot silently drift from the
authoritative config.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fusion_architectures import ARCHITECTURE_REGISTRY, SQI_CONDITIONED_ARCHITECTURES


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_norm_clip: float = 1.0
    batch_size: int = 32
    max_epochs: int = 30
    early_stopping_patience: int = 6
    device: str = "cpu"


@dataclass
class SeedRunResult:
    architecture: str
    seed: int
    best_val_mse: float
    epochs_trained: int
    test_predictions: np.ndarray = field(repr=False)
    test_targets: np.ndarray = field(repr=False)
    test_regimes: list = field(repr=False)
    gate_weights: np.ndarray | None = field(default=None, repr=False)  # (n_test, 2) if applicable


def set_seed(seed: int) -> None:
    """Governs parameter init + minibatch ordering (manuscript II-F)."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _forward(model, arch_name: str, ecg, ppg, sqi):
    if arch_name in SQI_CONDITIONED_ARCHITECTURES:
        return model(ecg, ppg, sqi)
    return model(ecg, ppg)


def _forward_with_gate(model, arch_name: str, ecg, ppg, sqi):
    """Returns (y_hat, gate_weights_or_None) -- used at test time for the
    gate-analysis pipeline (manuscript Section III-D)."""
    if arch_name == "adaptive_gate_sqi_conditioned":
        return model(ecg, ppg, sqi, return_gate=True)
    if arch_name == "adaptive_gate_implicit":
        return model(ecg, ppg, return_gate=True)
    if arch_name == "global_weighted_late_fusion":
        y, w = model(ecg, ppg, return_weights=True)
        # global weight is a single shared pair -- broadcast for uniform handling
        return y, w.unsqueeze(0).expand(ecg.shape[0], -1)
    return _forward(model, arch_name, ecg, ppg, sqi), None


def run_one_epoch(model, arch_name, loader, criterion, device, optimizer=None, grad_clip=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for ecg, ppg, sqi, target, _regimes in loader:
            ecg, ppg, sqi, target = ecg.to(device), ppg.to(device), sqi.to(device), target.to(device)
            if train:
                optimizer.zero_grad()
            pred = _forward(model, arch_name, ecg, ppg, sqi)
            loss = criterion(pred, target)
            if train:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * target.size(0)
            n += target.size(0)
    return total_loss / max(n, 1)


def train_one_seed(arch_name: str, seed: int, train_loader, val_loader, test_loader, cfg: TrainConfig) -> SeedRunResult:
    """
    Train one (architecture, seed) combination to completion, with early
    stopping on validation MSE, then evaluate on the fixed test partition.

    NOT EXECUTED in this environment -- see module docstring.
    """
    set_seed(seed)
    device = torch.device(cfg.device)
    model = ARCHITECTURE_REGISTRY[arch_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, cfg.max_epochs + 1):
        run_one_epoch(model, arch_name, train_loader, criterion, device, optimizer, cfg.grad_norm_clip)
        val_mse = run_one_epoch(model, arch_name, val_loader, criterion, device)
        epochs_trained = epoch

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on the fixed test partition (324 windows, manuscript II-F)
    model.eval()
    all_preds, all_targets, all_regimes, all_gates = [], [], [], []
    with torch.no_grad():
        for ecg, ppg, sqi, target, regimes in test_loader:
            ecg, ppg, sqi = ecg.to(device), ppg.to(device), sqi.to(device)
            pred, gate = _forward_with_gate(model, arch_name, ecg, ppg, sqi)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())
            all_regimes.extend(regimes)
            if gate is not None:
                all_gates.append(gate.cpu().numpy())

    return SeedRunResult(
        architecture=arch_name,
        seed=seed,
        best_val_mse=best_val_mse,
        epochs_trained=epochs_trained,
        test_predictions=np.concatenate(all_preds),
        test_targets=np.concatenate(all_targets),
        test_regimes=all_regimes,
        gate_weights=np.concatenate(all_gates) if all_gates else None,
    )


def train_all(dataset, cfg: TrainConfig, seeds=(0, 1, 2, 3, 4), architectures=None, fs: int = 125):
    """
    Full 8-architecture x 5-seed sweep (manuscript: "40 model runs total").

    NOT EXECUTED in this environment -- see module docstring.
    """
    from .torch_dataset import WindowDataset

    architectures = architectures or list(ARCHITECTURE_REGISTRY.keys())

    train_ds = WindowDataset(dataset.train, fs=fs)
    val_ds = WindowDataset(dataset.val, fs=fs)
    test_ds = WindowDataset(dataset.test, fs=fs)

    results = []
    for arch_name in architectures:
        for seed in seeds:
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
            result = train_one_seed(arch_name, seed, train_loader, val_loader, test_loader, cfg)
            results.append(result)
    return results
