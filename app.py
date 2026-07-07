from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from src.data import make_synthetic, create_dataloaders, LAYOUT_CLASSES, NUM_CLASSES
from src.model import build_layout_model
from src.core import compute_layout_metrics
from src.visualizations import plot_sample, plot_per_class_metrics, plot_training_history, CLASS_NAMES

st.set_page_config(page_title="DocIntel | Document Layout Analysis", layout="wide", page_icon="\U0001f4c4")

@st.cache_resource
def load_model():
    m = build_layout_model(num_classes=NUM_CLASSES)
    p = Path("models/best_model.pt")
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m

with st.sidebar:
    st.header("\u2699 Config")
    n_docs = st.slider("Documents", 100, 2000, 500, 50)
    show_preds = st.checkbox("Show Predictions", True)
    st.caption("DocIntel | Layout Analysis | Document AI")

data = make_synthetic(n=n_docs, seed=42)
_, val_loader = create_dataloaders(data, batch_size=4, seed=42)
val_images, val_masks = next(iter(val_loader))
model = load_model()

with torch.no_grad():
    val_logits = model(val_images)["out"]
    val_preds = val_logits.argmax(dim=1)

metrics = compute_layout_metrics(val_preds.numpy(), val_masks.numpy(), NUM_CLASSES)

c1, c2, c3, c4 = st.columns(4)
c1.metric("mIoU", f"{metrics['miou']:.4f}")
c2.metric("Pixel Acc", f"{metrics['pixel_acc']:.4f}")
c3.metric("Samples", f"{n_docs:,}")
c4.metric("Layout Classes", f"{NUM_CLASSES}")

t1, t2, t3, t4 = st.tabs(["\U0001f4ca Explorer", "\U0001f52c Model Lab", "\U0001f4d0 Layout Quality", "\U0001f4cb Document Types"])

with t1:
    st.subheader("Sample Layout Predictions")
    cols = st.columns(4)
    for i in range(min(4, len(val_images))):
        with cols[i]:
            fig = plot_sample(val_images[i], val_masks[i], val_preds[i] if show_preds else None)
            st.pyplot(fig)

with t2:
    rows = []
    for cls in range(NUM_CLASSES):
        rows.append({
            "Class": CLASS_NAMES[cls],
            "IoU": f"{metrics['per_class_iou'][cls]:.4f}",
        })
    st.dataframe(rows, use_container_width=True)
    st.pyplot(plot_per_class_metrics(metrics["per_class_iou"]))

with t3:
    st.subheader("Layout Segmentation Quality")
    st.latex(r"\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \quad \text{mIoU} = \frac{1}{C}\sum_{c=1}^C \text{IoU}_c")
    st.caption("Pixel-level Intersection over Union per layout class. Background excluded from mIoU in standard benchmarks.")
    st.latex(r"\text{Layout Loss} = \lambda_{CE} \cdot \mathcal{L}_{CE} + \lambda_{Dice} \cdot \mathcal{L}_{Dice}")
    st.caption("Combined Cross-Entropy + Dice loss handles class imbalance (background dominates).")
    col_a, col_b = st.columns(2)
    with col_a:
        from src.visualizations import _style; _style()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(NUM_CLASSES), metrics["per_class_iou"], color="#22d3ee", alpha=0.8)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45)
        ax.set_title("Per-Class IoU", color="white"); ax.grid(True, alpha=.2)
        st.pyplot(fig)
    with col_b:
        st.metric("Overall mIoU", f"{metrics['miou']:.4f}")
        st.metric("Pixel Accuracy", f"{metrics['pixel_acc']:.4f}")
        class_counts = [(val_masks == cls).sum().item() for cls in range(NUM_CLASSES)]
        total = sum(class_counts)
        for cls, count in enumerate(class_counts):
            st.caption(f"{CLASS_NAMES[cls]}: {count/total:.1%} of pixels")

with t4:
    st.subheader("Document Type Classification")
    st.info("Document type classification (invoice/receipt/contract/id_card/bank_statement/form) can be added as a multi-head extension to the layout model backbone.")
    st.latex(r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{layout}} + \lambda_{\text{doc}} \cdot \mathcal{L}_{\text{doc\_type}}")
    st.caption("Multi-task learning: shared backbone predicts both pixel-level layout and document-level category.")
