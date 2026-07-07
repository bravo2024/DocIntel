from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from src.data import make_synthetic, create_dataloaders, NUM_CLASSES
from src.model import build_layout_model
from src.core import compute_layout_metrics

def test_data():
    data = make_synthetic(n=20, seed=42)
    assert data["images"].shape == (20, 3, 224, 224)
    assert data["masks"].shape == (20, 224, 224)
    assert data["num_classes"] == 7

def test_dataloader():
    data = make_synthetic(n=20, seed=42)
    tl, vl = create_dataloaders(data, batch_size=8)
    batch = next(iter(tl))
    assert batch[0].shape == (8, 3, 224, 224)
    assert batch[1].shape == (8, 224, 224)

def test_model():
    model = build_layout_model(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)["out"]
    assert out.shape == (2, 7, 224, 224)

def test_metrics():
    pred = torch.randint(0, 7, (10, 224, 224)).numpy()
    true = torch.randint(0, 7, (10, 224, 224)).numpy()
    m = compute_layout_metrics(pred, true, 7)
    assert 0 <= m["miou"] <= 1
    assert 0 <= m["pixel_acc"] <= 1

def test_forward_pass():
    model = build_layout_model(num_classes=7)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))["out"]
    pred = out.argmax(dim=1)
    assert pred.shape == (1, 224, 224)
    assert pred.min() >= 0 and pred.max() < 7
