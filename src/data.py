from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMG_SIZE = 224
LAYOUT_CLASSES = ["background", "text", "table", "image", "signature", "barcode", "form_field"]
NUM_CLASSES = len(LAYOUT_CLASSES)

def _draw_text_block(img, mask, rng):
    h, w = img.shape[:2]
    for _ in range(rng.integers(2, 6)):
        bw = rng.integers(40, w - 20)
        bh = rng.integers(15, 40)
        bx = rng.integers(5, w - bw - 5)
        by = rng.integers(5, h - bh - 5)
        img[by:by+bh, bx:bx+bw] = (245, 245, 240)
        noise = rng.normal(0, 5, (bh, bw, 3)).astype(np.int16)
        img[by:by+bh, bx:bx+bw] = np.clip(img[by:by+bh, bx:bx+bw].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        mask[by:by+bh, bx:bx+bw] = 1

def _draw_table(img, mask, rng):
    h, w = img.shape[:2]
    for _ in range(rng.integers(0, 3)):
        tw = rng.integers(60, w - 20)
        th = rng.integers(40, 100)
        tx = rng.integers(5, w - tw - 5)
        ty = rng.integers(5, h - th - 5)
        for row in range(ty, ty + th, max(th // 5, 3)):
            img[row:row+2, tx:tx+tw] = (200, 200, 220)
            mask[row:row+2, tx:tx+tw] = 2
        for col in range(tx, tx + tw, max(tw // 4, 3)):
            img[ty:ty+th, col:col+1] = (200, 200, 220)
            mask[ty:ty+th, col:col+1] = 2
        cell_color = (235, 235, 245)
        img[ty+2:ty+th-2, tx+2:tx+tw-2][::max(th // 5, 3), :] = cell_color
        mask[ty+2:ty+th-2, tx+2:tx+tw-2][::max(th // 5, 3), :] = 2

def _draw_image_block(img, mask, rng):
    h, w = img.shape[:2]
    for _ in range(rng.integers(0, 3)):
        iw = rng.integers(30, 80)
        ih = rng.integers(30, 80)
        ix = rng.integers(5, w - iw - 5)
        iy = rng.integers(5, h - ih - 5)
        img[iy:iy+ih, ix:ix+iw] = rng.integers(50, 200, (ih, iw, 3), dtype=np.uint8)
        mask[iy:iy+ih, ix:ix+iw] = 3

def _draw_signature(img, mask, rng):
    h, w = img.shape[:2]
    for _ in range(rng.integers(0, 2)):
        sw = rng.integers(30, 80)
        sh = rng.integers(15, 30)
        sx = rng.integers(5, w - sw - 5)
        sy = rng.integers(h - 60, h - sh - 5)
        curve = np.sin(np.linspace(0, 4 * np.pi, sw)) * (sh // 4) + sh // 2
        for x in range(sw):
            y = int(sy + curve[x])
            if 0 <= y < h:
                img[y, sx + x] = (60, 60, 180)
                mask[y, sx + x] = 4

def _draw_barcode(img, mask, rng):
    h, w = img.shape[:2]
    for _ in range(rng.integers(0, 2)):
        bw = rng.integers(60, 120)
        bh = rng.integers(20, 30)
        bx = rng.integers(5, w - bw - 5)
        by = rng.integers(5, h - bh - 5)
        for x in range(bw):
            if rng.random() > 0.5:
                img[by:by+bh, bx+x] = (255, 255, 255)
                mask[by:by+bh, bx+x] = 5
            else:
                img[by:by+bh, bx+x] = (0, 0, 0)
                mask[by:by+bh, bx+x] = 5

def _draw_form_field(img, mask, rng):
    h, w = img.shape[:2]
    for _ in range(rng.integers(0, 4)):
        fw = rng.integers(30, 80)
        fh = rng.integers(12, 20)
        fx = rng.integers(5, w - fw - 5)
        fy = rng.integers(5, h - fh - 5)
        img[fy:fy+fh, fx:fx+fw] = (230, 230, 250)
        img[fy:fy+1, fx:fx+fw] = (100, 100, 200)
        img[fy+fh-1:fy+fh, fx:fx+fw] = (100, 100, 200)
        img[fy:fy+fh, fx:fx+1] = (100, 100, 200)
        img[fy:fy+fh, fx+fw-1:fx+fw] = (100, 100, 200)
        mask[fy:fy+fh, fx:fx+fw] = 6

def make_synthetic(n=500, img_size=IMG_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    images, masks, doc_types = [], [], []
    for _ in range(n):
        img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        mask = np.zeros((img_size, img_size), dtype=np.int64)
        _draw_text_block(img, mask, rng)
        _draw_table(img, mask, rng)
        _draw_image_block(img, mask, rng)
        _draw_signature(img, mask, rng)
        _draw_barcode(img, mask, rng)
        _draw_form_field(img, mask, rng)
        images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0)
        masks.append(torch.from_numpy(mask).long())
        doc_types.append(rng.integers(0, 6))
    data = {
        "images": torch.stack(images),
        "masks": torch.stack(masks),
        "doc_types": torch.tensor(doc_types),
        "class_names": LAYOUT_CLASSES,
        "num_classes": NUM_CLASSES,
        "n_samples": n,
    }
    return data

class LayoutDataset(Dataset):
    def __init__(self, data, split="train", val_split=0.2, seed=42):
        n = data["n_samples"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split_n = int(n * (1 - val_split))
        indices = idx[:split_n] if split == "train" else idx[split_n:]
        self.images = data["images"][indices]
        self.masks = data["masks"][indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

def create_dataloaders(data, batch_size=16, val_split=0.2, seed=42):
    train_ds = LayoutDataset(data, "train", val_split, seed)
    val_ds = LayoutDataset(data, "val", val_split, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
