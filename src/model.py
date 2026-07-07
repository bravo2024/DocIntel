from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
from src.core import LayoutLoss, compute_layout_metrics

def build_layout_model(num_classes=7):
    import torchvision
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    )
    model.classifier[4] = nn.Conv2d(model.classifier[4].in_channels, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(model.aux_classifier[4].in_channels, num_classes, kernel_size=1)
    return model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs["out"], masks)
            if "aux" in outputs:
                loss += 0.4 * criterion(outputs["aux"], masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_masks.append(masks.cpu().numpy())
    preds = np.concatenate(all_preds)
    masks = np.concatenate(all_masks)
    metrics = compute_layout_metrics(preds, masks, num_classes)
    metrics["loss"] = total_loss / len(loader)
    return metrics

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device="cuda", num_classes=7):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = LayoutLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = GradScaler()
    best_miou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_metrics["miou"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_miou"].append(val_metrics["miou"])
        history["val_acc"].append(val_metrics["pixel_acc"])
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"mIoU: {val_metrics['miou']:.4f} | PixAcc: {val_metrics['pixel_acc']:.4f}")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(model.state_dict(), Path("models/best_model.pt"))
            print(f"  -> Saved best model (mIoU={best_miou:.4f})")
    model.load_state_dict(torch.load(Path("models/best_model.pt")))
    history["best_miou"] = best_miou
    return model, history
