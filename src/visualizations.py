from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#1a1f2e", "figure.facecolor": "#1a1f2e",
        "axes.edgecolor": "#4a5568", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "text.color": "white", "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": "#4a5568",
    })

LAYOUT_COLORMAP = np.array([
    [0, 0, 0], [200, 200, 255], [200, 255, 200], [255, 200, 200],
    [200, 200, 100], [100, 100, 200], [255, 255, 150]
], dtype=np.uint8)

CLASS_NAMES = ["background", "text", "table", "image", "signature", "barcode", "form_field"]

def decode_layout(mask):
    return LAYOUT_COLORMAP[mask]

def plot_sample(image_tensor, mask_tensor, pred_tensor=None):
    _style()
    n = 2 if pred_tensor is None else 3
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    ax1.imshow(img); ax1.set_title("Document", color="white"); ax1.axis("off")
    gt_rgb = decode_layout(mask_tensor.cpu().numpy())
    ax2.imshow(gt_rgb); ax2.set_title("Layout GT", color="white"); ax2.axis("off")
    if pred_tensor is not None:
        pred_rgb = decode_layout(pred_tensor.cpu().numpy())
        ax3.imshow(pred_rgb); ax3.set_title("Layout Pred", color="white"); ax3.axis("off")
    plt.tight_layout()
    return fig

def plot_per_class_metrics(per_class_iou):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASS_NAMES))
    ax.bar(x, per_class_iou, color="#22d3ee", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_title("Per-Class IoU", color="white"); ax.grid(True, alpha=.2)
    plt.tight_layout()
    return fig

def plot_training_history(history):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train", color="#22d3ee")
    ax1.plot(epochs, history["val_loss"], label="Val", color="#f97316")
    ax1.set_title("Loss", color="white"); ax1.legend(); ax1.grid(True, alpha=.2)
    ax2.plot(epochs, history["val_miou"], color="#a78bfa")
    ax2.set_title("mIoU", color="white"); ax2.grid(True, alpha=.2)
    plt.tight_layout()
    return fig
