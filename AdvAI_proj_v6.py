# -*- coding: utf-8 -*-
"""
ViT-Small LoRA Change Detection Pipeline (v6 - Modular, Checkpointed & Visualizations)
Author: Shaheen / Gemini Notebook
Description:
  1. Reproducibility: Fully seeded random generation across Python, NumPy, PyTorch, CUDA.
  2. Checkpointing: Saves trained model weights (.pth) and metric logs (.json) to disk.
  3. Independent Visualization Phase: Load existing checkpoints and metrics instantly without retraining!
  4. Kaggle/Colab Ready: Switch batch_size (MX150=2, Kaggle/Colab=8 or 16).
  5. Visual Outputs:
     - Quantitative metric comparison charts (F1 vs Seasonal FAR)
     - 4-Panel Qualitative Prediction Grids (T1 | T2 Seasonal | Ground Truth | Prediction)
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import timm
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Reproducibility: Seed Configuration
# ==========================================
def set_seed(seed=42):
    """
    Ensures 100% deterministic reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Random seed fixed to: {seed}")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Loss & Metrics
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        probs = torch.sigmoid(outputs)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss.mean()

def compute_metrics(preds, targets, threshold=0.5):
    preds_binary = (torch.sigmoid(preds) > threshold).float()
    p = preds_binary.view(-1).cpu().numpy()
    t = targets.view(-1).cpu().numpy()
    
    intersection = np.logical_and(p == 1, t == 1).sum()
    union = np.logical_or(p == 1, t == 1).sum()
    
    tp = intersection
    fp = np.logical_and(p == 1, t == 0).sum()
    fn = np.logical_and(p == 0, t == 1).sum()
    tn = np.logical_and(p == 0, t == 0).sum()
    
    epsilon = 1e-7
    f1 = (2 * tp) / (2 * tp + fp + fn + epsilon)
    iou = tp / (union + epsilon)
    far = fp / (fp + tn + epsilon)
    return float(f1), float(iou), float(far)

# ==========================================
# 3. Dataset & Spatial Split
# ==========================================
class MockChangeDetectionDataset(Dataset):
    def __init__(self, num_samples=200, img_size=224, is_seasonal=False):
        self.num_samples = num_samples
        self.img_size = img_size
        self.is_seasonal = is_seasonal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Deterministic generation per index
        g = torch.Generator().manual_seed(idx + (9999 if self.is_seasonal else 0))
        t1 = torch.randn(3, self.img_size, self.img_size, generator=g)
        noise_level = 0.4 if self.is_seasonal else 0.1
        t2 = t1 + noise_level * torch.randn(3, self.img_size, self.img_size, generator=g)
        mask = torch.randint(0, 2, (1, self.img_size, self.img_size), generator=g).float()
        return t1, t2, mask

def spatial_block_split(dataset, grid_rows=4, grid_cols=4, test_ratio=0.5, seed=42):
    np.random.seed(seed)
    total_blocks = grid_rows * grid_cols
    block_ids = np.arange(total_blocks)
    np.random.shuffle(block_ids)
    num_test_blocks = int(total_blocks * test_ratio)
    test_blocks = set(block_ids[:num_test_blocks])
    
    train_indices, test_indices = [], []
    for idx in range(len(dataset)):
        if (idx % total_blocks) in test_blocks:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    return Subset(dataset, train_indices), Subset(dataset, test_indices)

# ==========================================
# 4. Model Architectures
# ==========================================
class SiameseUNet(nn.Module):
    def __init__(self):
        super(SiameseUNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, t1, t2):
        f1 = self.encoder(t1)
        f2 = self.encoder(t2)
        return self.decoder(torch.abs(f1 - f2))

class FrozenViTChangeDetector(nn.Module):
    def __init__(self):
        super(FrozenViTChangeDetector, self).__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def extract_feature_map(self, x):
        feats = self.backbone.forward_features(x)
        patch_tokens = feats[:, 1:, :]
        B, N, C = patch_tokens.shape
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, 14, 14)
        return nn.functional.interpolate(feature_map, size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, t1, t2):
        f1 = self.extract_feature_map(t1)
        f2 = self.extract_feature_map(t2)
        return self.head(torch.abs(f1 - f2))

class ViTChangeDetector(nn.Module):
    def __init__(self):
        super(ViTChangeDetector, self).__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.1, bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.head = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def extract_feature_map(self, x):
        feats = self.backbone.forward_features(x)
        patch_tokens = feats[:, 1:, :]
        B, N, C = patch_tokens.shape
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, 14, 14)
        return nn.functional.interpolate(feature_map, size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, t1, t2):
        f1 = self.extract_feature_map(t1)
        f2 = self.extract_feature_map(t2)
        return self.head(torch.abs(f1 - f2))

# ==========================================
# 5. Checkpoint Saving & Loading Functions
# ==========================================
CHECKPOINT_DIR = "./checkpoints"

def save_model_checkpoint(model, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(model.state_dict(), filepath)
    print(f"  ✓ Saved checkpoint: {filepath}")

def load_model_checkpoint(model, filename):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"  ✓ Loaded model weights from: {filepath}")
        return True
    else:
        print(f"  ⚠️ Warning: Checkpoint file {filepath} not found.")
        return False

def save_metrics_json(metrics_dict, filename="experiment_results.json"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"  ✓ Saved quantitative metrics to: {filepath}")

def load_metrics_json(filename="experiment_results.json"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)
        print(f"  ✓ Loaded saved metrics from: {filepath}")
        return metrics_dict
    return None

# ==========================================
# 6. Training & Evaluation Pipeline
# ==========================================
def train_and_save_pipeline(batch_size=2, epochs=1):
    set_seed(42)
    print("\n==================================================")
    print("=== Training & Checkpointing Pipeline ===")
    print(f"Batch Size: {batch_size} | Device: {device}")
    print("==================================================\n")

    full_dataset = MockChangeDetectionDataset(num_samples=200, is_seasonal=False)
    seasonal_dataset = MockChangeDetectionDataset(num_samples=50, is_seasonal=True)
    train_set, test_set = spatial_block_split(full_dataset, test_ratio=0.5, seed=42)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    seasonal_loader = DataLoader(seasonal_dataset, batch_size=batch_size, shuffle=False)

    models_info = [
        ("Siamese U-Net (ResNet18)", SiameseUNet(), "siamese_unet.pth"),
        ("Frozen ViT (No LoRA)", FrozenViTChangeDetector(), "frozen_vit.pth"),
        ("Frozen ViT + LoRA (Proposed)", ViTChangeDetector(), "vit_lora.pth")
    ]

    results = {}
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for name, model, ckpt_name in models_info:
        print(f"\n---> Training Model: {name}")
        model = model.to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        criterion = BCEDiceLoss(bce_weight=0.5)

        model.train()
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", unit="batch")
            running_loss = 0.0
            for t1, t2, mask in pbar:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(t1, t2)
                    loss = criterion(outputs, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                pbar.set_postfix({'avg_loss': f"{running_loss / len(train_loader):.4f}"})

        # Save checkpoint
        save_model_checkpoint(model, ckpt_name)

        # Evaluation
        model.eval()
        gen_f1, gen_iou, gen_far = [], [], []
        with torch.no_grad():
            for t1, t2, mask in test_loader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                out = model(t1, t2)
                f1, iou, far = compute_metrics(out, mask)
                gen_f1.append(f1); gen_iou.append(iou); gen_far.append(far)

        seas_f1, seas_iou, seas_far = [], [], []
        with torch.no_grad():
            for t1, t2, mask in seasonal_loader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                out = model(t1, t2)
                f1, iou, far = compute_metrics(out, mask)
                seas_f1.append(f1); seas_iou.append(iou); seas_far.append(far)

        results[name] = {
            'General_F1': float(np.mean(gen_f1)),
            'General_IoU': float(np.mean(gen_iou)),
            'General_FAR': float(np.mean(gen_far)),
            'Seasonal_FAR': float(np.mean(seas_far))
        }

    save_metrics_json(results)
    return results

# ==========================================
# 7. Visualizations (No Retraining Required!)
# ==========================================
def plot_quantitative_comparison(results=None):
    if results is None:
        results = load_metrics_json()
        if results is None:
            print("❌ No saved metrics found! Run training pipeline first.")
            return

    model_names = list(results.keys())
    gen_f1 = [results[m]['General_F1'] for m in model_names]
    seas_far = [results[m]['Seasonal_FAR'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    rects1 = ax1.bar(x - width/2, gen_f1, width, label='General F1-Score (Higher is better)', color='skyblue')
    ax1.set_ylabel('F1 Score', color='blue')
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, seas_far, width, label='Seasonal False Alarm Rate (Lower is better)', color='salmon')
    ax2.set_ylabel('Seasonal FAR', color='red')
    ax2.set_ylim(0, 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=10, fontsize=9)
    plt.title("Quantitative Comparison: Accuracy (F1) vs. Seasonal False Alarm Suppression (FAR)")
    fig.tight_layout()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/quantitative_comparison.png", dpi=300)
    print("  ✓ Saved plot: ./plots/quantitative_comparison.png")
    plt.show()

def plot_qualitative_4panel(batch_size=2, num_samples=3):
    set_seed(42)
    print("\n---> Generating Qualitative 4-Panel Prediction Maps...")
    
    seasonal_dataset = MockChangeDetectionDataset(num_samples=num_samples, is_seasonal=True)
    loader = DataLoader(seasonal_dataset, batch_size=1, shuffle=False)

    models_info = [
        ("Siamese U-Net", SiameseUNet(), "siamese_unet.pth"),
        ("Frozen ViT (No LoRA)", FrozenViTChangeDetector(), "frozen_vit.pth"),
        ("Frozen ViT + LoRA (Proposed)", ViTChangeDetector(), "vit_lora.pth")
    ]

    # Load checkpoints
    loaded_models = []
    for name, model, ckpt_name in models_info:
        model = model.to(device)
        load_model_checkpoint(model, ckpt_name)
        model.eval()
        loaded_models.append((name, model))

    for sample_idx, (t1_img, t2_img, mask) in enumerate(loader):
        t1_img, t2_img, mask = t1_img.to(device), t2_img.to(device), mask.to(device)
        
        t1_np = np.clip(t1_img[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
        t2_np = np.clip(t2_img[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
        gt_np = mask[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 2 + len(loaded_models), figsize=(16, 4))
        
        axes[0].imshow(t1_np)
        axes[0].set_title("T1 Image (Pre)", fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(t2_np)
        axes[1].set_title("T2 Image (Seasonal Shift)", fontsize=10)
        axes[1].axis('off')

        axes[2].imshow(gt_np, cmap='gray')
        axes[2].set_title("Ground Truth", fontsize=10)
        axes[2].axis('off')

        with torch.no_grad():
            for col_idx, (m_name, model) in enumerate(loaded_models):
                out = model(t1_img, t2_img)
                pred = (torch.sigmoid(out) > 0.5).float()[0, 0].cpu().numpy()
                ax_idx = 3 + col_idx
                if ax_idx < len(axes):
                    axes[ax_idx].imshow(pred, cmap='gray')
                    axes[ax_idx].set_title(m_name, fontsize=10)
                    axes[ax_idx].axis('off')

        plt.suptitle(f"Qualitative Prediction Comparison - Sample #{sample_idx+1}", fontsize=12, y=1.02)
        fig.tight_layout()
        os.makedirs("./plots", exist_ok=True)
        save_path = f"./plots/qualitative_comparison_sample_{sample_idx+1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved qualitative grid: {save_path}")
        plt.show()

# ==========================================
# 8. Main Execution Switch
# ==========================================
if __name__ == "__main__":
    # MODE 1: Train and Save Checkpoints + Metrics (Run this ONCE or when retrain is needed)
    # results = train_and_save_pipeline(batch_size=2, epochs=1)

    # MODE 2: Visualizations Only (Loads existing checkpoints and metrics instantly!)
    print("=== Running Visualization Phase (Without Retraining) ===")
    plot_quantitative_comparison()
    # plot_qualitative_4panel(num_samples=2)
