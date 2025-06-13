#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THY classification – train / val / test 三資料夾版
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # 靜音 TensorFlow
import timm, torch, matplotlib
matplotlib.use('Agg')                             # 無 GUI 環境仍可存圖
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
    ConfusionMatrixDisplay
)
from multiprocessing import freeze_support
from typing import List


# ---------- 繪圖輔助 ----------
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig


def plot_roc_curves(one_hot_labels: np.ndarray, probs: np.ndarray,
                    class_names: List[str], title: str) -> (plt.Figure, float):
    n_classes = one_hot_labels.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], probs[:, i])
        roc_auc[i] = sk_auc(fpr[i], tpr[i])

    # macro-average AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = sk_auc(all_fpr, mean_tpr)

    fig = plt.figure(figsize=(6, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} AUC={roc_auc[i]:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig, macro_auc


# ---------- 自訂 ImageFolder，排除隱藏資料夾 ----------
class CleanImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory)
                   if d.is_dir() and not d.name.startswith('.')]
        if not classes:
            raise FileNotFoundError(f"❌ 找不到合法類別資料夾於 {directory}")
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def main():
    freeze_support()

    # ======== 參數區 ========
    train_root = r"THY/train"       # 修改成你的實際路徑
    val_root   = r"THY/val"
    test_root  = r"THY/test"
    checkpoint_dir = "THY_Classification_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = 64
    max_epochs = 100
    min_epochs = 20
    earlystop_patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_train = [
        "resnet50",
        "efficientnet_b0",
        "vit_base_resnet50d_224",
        "swinv2_small_window8_256",
        "pit_ti_224",
        "efficientnetv2_s",
        "regnety_004",
        "efficientformer_l1",
        "resnest50d"
    ]

    # ======== 共用前處理（基準 224×224）========
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 載入三份資料集
    train_dataset = CleanImageFolder(train_root, transform=base_transform)
    val_dataset   = CleanImageFolder(val_root,   transform=base_transform)
    test_dataset  = CleanImageFolder(test_root,  transform=base_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # 計算 class weights（給交叉熵）
    from collections import Counter
    label_counts = Counter([lbl for _, lbl in train_dataset])
    total_samples = sum(label_counts.values())
    class_weights = [
        total_samples / (num_classes * label_counts[i])
        for i in range(num_classes)
    ]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 預先建立 dataloader（transform 每回合可被覆蓋）
    def make_loaders(ds_train, ds_val, ds_test, bs=64):
        train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(ds_val,   batch_size=bs, shuffle=False, num_workers=2)
        test_loader  = DataLoader(ds_test,  batch_size=bs, shuffle=False, num_workers=2)
        return train_loader, val_loader, test_loader

    results = []

    for model_name in models_to_train:
        print(f"\n===== Training {model_name} =====")

        # 動態決定輸入大小
        input_size = 256 if "swin" in model_name else 224
        cur_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # 替換 dataset 內的 transform
        train_dataset.transform = cur_transform
        val_dataset.transform   = cur_transform
        test_dataset.transform  = cur_transform

        train_loader, val_loader, test_loader = make_loaders(
            train_dataset, val_dataset, test_dataset, batch_size)

        # 建立 / 載入模型
        try:
            model = timm.create_model(
                model_name, pretrained=False, num_classes=num_classes).to(device)
        except Exception as e:
            print(f"模型 {model_name} 載入失敗：{e}")
            continue

        criterion  = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer  = Adam(model.parameters(), lr=1e-3)
        scheduler  = StepLR(optimizer, step_size=3, gamma=0.1)

        best_val_acc    = 0.0
        best_model_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        epochs_no_improve = 0

        # TensorBoard
        log_dir = os.path.join("runs", "THY", model_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # --------- 迴圈 ----------
        for epoch in range(max_epochs):
            # --- Train ---
            model.train()
            running_loss, running_correct = 0.0, 0
            for inputs, labels in tqdm(train_loader,
                                       desc=f"{model_name} Epoch {epoch+1}/{max_epochs}",
                                       leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss   += loss.item() * inputs.size(0)
                running_correct += outputs.argmax(1).eq(labels).sum().item()

            train_loss = running_loss / len(train_dataset)
            train_acc  = running_correct / len(train_dataset)

            # --- Validation ---
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_correct += outputs.argmax(1).eq(labels).sum().item()
            val_acc = val_correct / len(val_dataset)

            # --- TensorBoard ---
            writer.add_scalar("Loss/train",  train_loss, epoch)
            writer.add_scalar("Acc/train",   train_acc,  epoch)
            writer.add_scalar("Acc/val",     val_acc,    epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

            # --- Early-stopping & checkpoint ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch+1}: train-loss {train_loss:.4f} | "
                  f"train-acc {train_acc:.4f} | val-acc {val_acc:.4f}")

            if epoch + 1 >= min_epochs and epochs_no_improve >= earlystop_patience:
                print("Early stopping triggered.")
                break

            scheduler.step()

        writer.close()

        # --------- Test ---------
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(probs.argmax(1).cpu())
                all_labels.append(labels.cpu())

        all_probs  = torch.cat(all_probs).numpy()
        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0)
        one_hot_labels = np.eye(num_classes)[all_labels]
        auc_score = roc_auc_score(one_hot_labels, all_probs,
                                  average="macro", multi_class="ovr")

        # Confusion Matrix
        cm_fig = ConfusionMatrixDisplay.from_predictions(
            all_labels, all_preds, cmap="Blues").figure_
        cm_fig.savefig(os.path.join(checkpoint_dir,
                                    f"{model_name}_confusion_matrix.png"))
        plt.close(cm_fig)

        # ROC
        roc_fig, _ = plot_roc_curves(one_hot_labels, all_probs,
                                     class_names, f"{model_name} ROC")
        roc_fig.savefig(os.path.join(checkpoint_dir,
                                     f"{model_name}_roc.png"))
        plt.close(roc_fig)

        # Save metrics
        results.append(dict(
            model=model_name,
            best_val_acc=best_val_acc,
            test_acc=acc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc=auc_score
        ))

    # -------- 匯出總結 --------
    pd.DataFrame(results).to_csv(
        os.path.join(checkpoint_dir, "THY_Classification_result.csv"),
        index=False)
    print("\n✅ Training complete. Results saved.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到可用 GPU，請確認 CUDA 驅動與 PyTorch CUDA 版本。")
    print(f"✅ 使用 GPU：{torch.cuda.get_device_name(0)}")
    main()
