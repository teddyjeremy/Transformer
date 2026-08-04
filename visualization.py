from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history, save_path=None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss")

    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    return fig, ax


def plot_validation_metrics(history, save_path=None):
    metrics = [
        key for key in ("cer", "wer", "bleu")
        if history.get(key)
    ]

    if not metrics:
        raise ValueError("No validation metrics were provided")

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, 4 * len(metrics)),
        squeeze=False
    )

    epochs = np.arange(1, len(history[metrics[0]]) + 1)

    for index, metric in enumerate(metrics):
        ax = axes[index, 0]
        ax.plot(epochs, history[metric])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Validation {metric.upper()}")
        ax.grid(alpha=0.25)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    return fig, axes


def plot_attention(
    attention,
    source_tokens,
    target_tokens,
    head=0,
    title="Attention",
    save_path=None
):
    attention = torch.as_tensor(attention).detach().cpu()

    if attention.dim() == 4:
        attention = attention[0]

    if attention.dim() != 3:
        raise ValueError("Attention must have shape (heads, query, key)")

    if head >= attention.size(0):
        raise ValueError("Attention head is out of range")

    matrix = attention[head].numpy()
    source_tokens = list(source_tokens)[:matrix.shape[1]]
    target_tokens = list(target_tokens)[:matrix.shape[0]]

    fig, ax = plt.subplots(
        figsize=(max(8, len(source_tokens) * 0.45), max(6, len(target_tokens) * 0.45))
    )
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=60, ha="right")
    ax.set_yticklabels(target_tokens)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Target tokens")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    return fig, ax


def plot_translation_lengths(
    source_lengths,
    target_lengths,
    save_path=None
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(source_lengths, bins=30, alpha=0.6, label="Source")
    ax.hist(target_lengths, bins=30, alpha=0.6, label="Target")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Examples")
    ax.set_title("Token Length Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    return fig, ax
