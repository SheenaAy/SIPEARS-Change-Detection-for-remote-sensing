# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:00:00 2026

Description of updates in this version:
    1. Updated cell blocks to ensure variable consistency across cells
    2. Cell 1 and 3 are the same as V1, rest have been updated
    
    
Change Log:
    
Cell 2: Loss Functions & Evaluation Metrics

• Removed: Your existing compute_metrics function.
• Added: compute_confusion_matrix below to return raw tp, fp, fn, tn counts per batch.


Cell 4: ViT-Small + LoRA Change Detection Model Architecture

• Removed: The entire old ViTChangeDetector class (which had the broken self.backbone.forward_features(x) call, redundant freezing loop, and early interpolate bottleneck).
• Added: The updated ViTChangeDetector class with .get_base_model().forward_features(x) and the progressive ConvTranspose2d decoder.


Cell 5: 10-Iteration 50/50 Split Training & Evaluation Loop

• Removed: iter_f1, iter_iou, iter_far lists and batch-averaging inside the evaluation block.
• Added: Dataset-wide total_tp, total_fp, total_fn, total_tn accumulation.


Cell 6: Quantitative Visualization

• Removed: labels=labels in plt.boxplot().
• Added: tick_labels=labels.


Cell 7: Qualitative Visualization

• Removed: t1 = np.clip(t1, 0, 1) and t2 = np.clip(t2, 0, 1).
• Added: Min-max scaling for clean visualization and uncommented the function call at the bottom (also fixed the axes indexing).

Runtime Expectations:
With the corrected ConvTranspose2d decoder (which runs at 14×14 instead of 224×224), 
the script runs 10 iterations × 1 epoch × 100 training images + 100 test images (2,000 forward passes total).

@author: Shaheen
"""

#%% Cell 1. Imports & Device Configuration 
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from peft import LoraConfig, get_peft_model
import numpy as np

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% Cell 2. Loss Functions & Evaluation Metrics
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        
        # Soft Dice Loss
        probs = torch.sigmoid(outputs)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()
        
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def compute_confusion_matrix(preds, targets, threshold=0.5):
    preds_binary = (torch.sigmoid(preds) > threshold).float()
    
    p = preds_binary.view(-1).cpu().numpy()
    t = targets.view(-1).cpu().numpy()
    
    tp = np.logical_and(p == 1, t == 1).sum()
    fp = np.logical_and(p == 1, t == 0).sum()
    fn = np.logical_and(p == 0, t == 1).sum()
    tn = np.logical_and(p == 0, t == 0).sum()
    
    return tp, fp, fn, tn


#%% Cell 3: Mock Remote Sensing Change Detection Dataset
class MockChangeDetectionDataset(Dataset):
    """
    Replace this with your actual LEVIR-CD / WHU-CD dataset class 
    that loads pairs of T1 and T2 images and a binary change mask.
    """
    def __init__(self, num_samples=200, img_size=224):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # T1 image: 3 channels, 224x224
        t1 = torch.randn(3, self.img_size, self.img_size)
        # T2 image: 3 channels, 224x224 (with some simulated changes)
        t2 = t1 + 0.1 * torch.randn(3, self.img_size, self.img_size)
        # Binary change mask: 1 channel, 224x224
        mask = torch.randint(0, 2, (1, self.img_size, self.img_size)).float()
        return t1, t2, mask

#%% Cell 4: ViT-Small + LoRA Change Detection Model Architecture
class ViTChangeDetector(nn.Module):
    def __init__(self):
        super(ViTChangeDetector, self).__init__()
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone = get_peft_model(backbone, lora_config)
        
        # Progressive Upsampling Decoder (14x14 -> 28 -> 56 -> 112 -> 224)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(24, 1, kernel_size=1)
        )

    def extract_feature_map(self, x):
        # Access underlying timm model so PEFT routes LoRA adapters properly
        feats = self.backbone.get_base_model().forward_features(x)
        patch_tokens = feats[:, 1:, :]  # Exclude class token -> (B, 196, 384)
        B, N, C = patch_tokens.shape
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, 14, 14)
        return feature_map

    def forward(self, t1, t2):
        feat1 = self.extract_feature_map(t1)
        feat2 = self.extract_feature_map(t2)
        diff_feat = torch.abs(feat1 - feat2)
        out = self.decoder(diff_feat)
        return out


#%% Cell 5: 10-Iteration 50/50 Split Training & Evaluation Loop
full_dataset = MockChangeDetectionDataset(num_samples=200)
total_size = len(full_dataset)
train_size = int(0.5 * total_size)
test_size = total_size - train_size

num_iterations = 10
all_f1, all_iou, all_far = [], [], []

print(f"Starting {num_iterations}-iteration experiment with 50/50 split...")

for iteration in range(num_iterations):
    print(f"\n================ Iteration {iteration + 1} / {num_iterations} ================")
    
    train_set, test_set = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    
    model = ViTChangeDetector().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = BCEDiceLoss(bce_weight=0.5)
    
    # Training
    model.train()
    for t1_img, t2_img, masks in train_loader:
        t1_img, t2_img, masks = t1_img.to(device), t2_img.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(t1_img, t2_img)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
    # Evaluation: Accumulate TP, FP, FN, TN across the entire test set
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for t1_img, t2_img, masks in test_loader:
            t1_img, t2_img, masks = t1_img.to(device), t2_img.to(device), masks.to(device)
            outputs = model(t1_img, t2_img)
            
            tp, fp, fn, tn = compute_confusion_matrix(outputs, masks)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
    epsilon = 1e-7
    iter_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)
    iter_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    iter_far = total_fp / (total_fp + total_tn + epsilon)
    
    all_f1.append(iter_f1)
    all_iou.append(iter_iou)
    all_far.append(iter_far)
    
    print(f"Result -> F1: {iter_f1:.4f} | IoU: {iter_iou:.4f} | FAR: {iter_far:.4f}")

print("\n================ Final Summary Across 10 Iterations ================")
print(f"Average F1-Score : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"Average IoU      : {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
print(f"Average FAR      : {np.mean(all_far):.4f} ± {np.std(all_far):.4f}")


#%% Cell 6: Quantitative Visualization
import matplotlib.pyplot as plt
import numpy as np

metrics_data = [all_f1, all_iou, all_far]
labels = ['F1 Score', 'IoU', 'False Alarm Rate (FAR)']

plt.figure(figsize=(8, 5))
plt.boxplot(
    metrics_data,
    tick_labels=labels,
    patch_artist=True,
    boxprops=dict(facecolor='lightblue', color='blue'),
    medianprops=dict(color='red', linewidth=2)
)
plt.title('10-Iteration Cross-Validation Performance (Frozen ViT + LoRA)')
plt.ylabel('Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



#%% Cell 7: Qualitative Visualization
def visualize_predictions(model, test_loader, device, num_samples=3):
    model.eval()
    count = 0
    with torch.no_grad():
        for t1_img, t2_img, masks in test_loader:
            t1_img, t2_img, masks = t1_img.to(device), t2_img.to(device), masks.to(device)
            outputs = model(t1_img, t2_img)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            for i in range(t1_img.size(0)):
                if count >= num_samples:
                    return
                
                t1 = t1_img[i].cpu().permute(1, 2, 0).numpy()
                t2 = t2_img[i].cpu().permute(1, 2, 0).numpy()
                gt = masks[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()
                
                # Min-max scale to [0, 1] for clean display of mock normal tensors
                t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)
                t2 = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(t1)
                axes[0].set_title("T1 Image")
                axes[0].axis('off')
                
                axes[1].imshow(t2)
                axes[1].set_title("T2 Image (Seasonal Shift)")
                axes[1].axis('off')
                
                axes[2].imshow(gt, cmap='gray')
                axes[2].set_title("Ground Truth Change")
                axes[2].axis('off')
                
                axes[3].imshow(pred, cmap='gray')
                axes[3].set_title("Model Prediction (LoRA)")
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.show()
                count += 1

visualize_predictions(model, test_loader, device, num_samples=3)


# Call this after training the model on the test loader:
# visualize_predictions(model, test_loader, device, num_samples=3)