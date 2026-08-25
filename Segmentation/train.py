import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from dataset import (
    get_train_dataset,
    get_val_dataset,
)
from model import create_model

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """
    def __init__(
        self,
        smooth=1.0,
    ):
        super().__init__()
        self.smooth = smooth
    def forward(
        self,
        logits,
        targets,
    ):
        probabilities = torch.sigmoid(
            logits
        )
        probabilities = probabilities.view(
            probabilities.size(0),
            -1,
        )
        targets = targets.view(
            targets.size(0),
            -1,
        )
        intersection = (
            probabilities * targets
        ).sum(dim=1)
        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )
        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )
        return (
            1.0 - dice
        ).mean()

class CombinedLoss(nn.Module):
    """
    BCE + Dice loss.
    """
    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(
        self,
        logits,
        targets,
    ):
        bce_loss = self.bce(
            logits,
            targets,
        )
        dice_loss = self.dice(
            logits,
            targets,
        )
        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )
        return total_loss

def create_confusion():
    return {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }

def update_confusion(
    predictions,
    targets,
    confusion,
):
    predictions = predictions.bool()
    targets = targets.bool()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    confusion["TP"] += int(
        (predictions & targets).sum().item()
    )
    confusion["TN"] += int(
        ((~predictions) & (~targets))
        .sum()
        .item()
    )
    confusion["FP"] += int(
        (predictions & (~targets))
        .sum()
        .item()
    )
    confusion["FN"] += int(
        ((~predictions) & targets)
        .sum()
        .item()
    )

def calculate_metrics(
    confusion,
):
    tp = confusion["TP"]
    tn = confusion["TN"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    epsilon = 1e-8
    iou = (
        tp
        / (
            tp + fp + fn + epsilon
        )
    )
    dice = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + epsilon
        )
    )
    pixel_accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + epsilon
    )
    precision = (
        tp
        / (
            tp + fp + epsilon
        )
    )
    recall = (
        tp
        / (
            tp + fn + epsilon
        )
    )
    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + epsilon
        )
    )
    background_iou = (
        tn
        / (
            tn + fp + fn + epsilon
        )
    )
    mean_iou = (
        iou + background_iou
    ) / 2.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": float(
            pixel_accuracy
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mean_iou": float(mean_iou),
        "background_iou": float(
            background_iou
        ),
        "crack_iou": float(iou),
    }

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):
    model.train()
    running_loss = 0.0
    confusion = create_confusion()
    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )
    for images, masks, _ in progress:
        images = images.to(
            config.DEVICE,
            non_blocking=True,
        )
        masks = masks.to(
            config.DEVICE,
            non_blocking=True,
        )
        optimizer.zero_grad(
            set_to_none=True
        )
        logits = model(
            images
        )
        loss = criterion(
            logits,
            masks,
        )
        loss.backward()
        optimizer.step()
        running_loss += (
            loss.item()
            * images.size(0)
        )
        probabilities = torch.sigmoid(
            logits
        )
        predictions = (
            probabilities
            >= config.PREDICTION_THRESHOLD
        )
        update_confusion(
            predictions,
            masks >= 0.5,
            confusion,
        )
        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )
    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )
    metrics = calculate_metrics(
        confusion
    )
    metrics["loss"] = float(
        epoch_loss
    )
    return metrics

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
):
    model.eval()
    running_loss = 0.0
    confusion = create_confusion()
    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )
    for images, masks, _ in progress:
        images = images.to(
            config.DEVICE,
            non_blocking=True,
        )
        masks = masks.to(
            config.DEVICE,
            non_blocking=True,
        )
        logits = model(
            images
        )
        loss = criterion(
            logits,
            masks,
        )
        running_loss += (
            loss.item()
            * images.size(0)
        )
        probabilities = torch.sigmoid(
            logits
        )
        predictions = (
            probabilities
            >= config.PREDICTION_THRESHOLD
        )
        update_confusion(
            predictions,
            masks >= 0.5,
            confusion,
        )
    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )
    metrics = calculate_metrics(
        confusion
    )
    metrics["loss"] = float(
        epoch_loss
    )
    return metrics

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "best_val_loss": (
            best_val_loss
        ),
    }
    torch.save(
        checkpoint,
        path,
    )

def save_history(history):
    config.TRAINING_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        config.TRAINING_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )

def save_history(history):
    config.TRAINING_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        config.TRAINING_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )
def train():
    print("\n" + "=" * 80)
    print(
        "U-NET + RESNET34 CRACK SEGMENTATION TRAINING"
    )
    print("=" * 80)
    config.create_directories()
    config.validate_dataset_directories()
    config.print_configuration()
    set_seed(
        config.RANDOM_SEED
    )
    # Dataset
    print(
        "\nLoading pre-split datasets..."
    )
    train_dataset = (
        get_train_dataset()
    )
    val_dataset = (
        get_val_dataset()
    )
    print(
        f"Training samples   : "
        f"{len(train_dataset):,}"
    )
    print(
        f"Validation samples : "
        f"{len(val_dataset):,}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )
    print(
        "\nCreating U-Net + ResNet34 model..."
    )
    model = create_model(
        pretrained=config.PRETRAINED_ENCODER
    )
    model = model.to(
        config.DEVICE
    )
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_pixel_accuracy": [],
        "val_pixel_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1_score": [],
        "val_f1_score": [],
        "train_mean_iou": [],
        "val_mean_iou": [],
        "train_background_iou": [],
        "val_background_iou": [],
        "train_crack_iou": [],
        "val_crack_iou": [],
        "learning_rate": [],
    }
    best_val_loss = float(
        "inf"
    )
    epochs_without_improvement = 0
    for epoch in range(
        1,
        config.NUM_EPOCHS + 1,
    ):
        print(
            "\n" + "=" * 80
        )
        print(
            f"Epoch "
            f"{epoch}/{config.NUM_EPOCHS}"
        )
        print(
            "=" * 80
        )
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
        )
        current_lr = (
            optimizer.param_groups[0]["lr"]
        )
        scheduler.step(
            val_metrics["loss"]
        )
        history["epoch"].append(
            epoch
        )
        history["train_loss"].append(
            train_metrics["loss"]
        )
        history["val_loss"].append(
            val_metrics["loss"]
        )
        for metric_name in [
            "iou",
            "dice",
            "pixel_accuracy",
            "precision",
            "recall",
            "f1_score",
            "mean_iou",
            "background_iou",
            "crack_iou",
        ]:
            history[
                f"train_{metric_name}"
            ].append(
                train_metrics[
                    metric_name
                ]
            )
            history[
                f"val_{metric_name}"
            ].append(
                val_metrics[
                    metric_name
                ]
            )
        history[
            "learning_rate"
        ].append(
            current_lr
        )
        save_history(
            history
        )
print("\nTraining:")
        print(
            f"  Loss      : "
            f"{train_metrics['loss']:.6f}"
        )
        print(
            f"  IoU       : "
            f"{train_metrics['iou']:.4f}"
        )
        print(
            f"  Dice      : "
            f"{train_metrics['dice']:.4f}"
        )
        print(
            f"  Precision : "
            f"{train_metrics['precision']:.4f}"
        )
        print(
            f"  Recall    : "
            f"{train_metrics['recall']:.4f}"
        )
        print("\nValidation:")
        print(
            f"  Loss      : "
            f"{val_metrics['loss']:.6f}"
        )
        print(
            f"  IoU       : "
            f"{val_metrics['iou']:.4f}"
        )
        print(
            f"  Dice      : "
            f"{val_metrics['dice']:.4f}"
        )
        print(
            f"  Precision : "
            f"{val_metrics['precision']:.4f}"
        )
        print(
            f"  Recall    : "
            f"{val_metrics['recall']:.4f}"
        )
        print(
            f"\nLearning rate: "
            f"{current_lr:.8f}"
        )
        save_checkpoint(
            path=config.LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
        )
        if (
            val_metrics["loss"]
            < best_val_loss
        ):
            best_val_loss = (
                val_metrics["loss"]
            )
            epochs_without_improvement = 0
            save_checkpoint(
                path=config.BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
            print(
                "\n*** Best model updated ***"
            )
        else:
            epochs_without_improvement += 1
            print(
                "\nNo validation-loss improvement."
            )
            
            print(
                f"Early stopping counter: "
                f"{epochs_without_improvement}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )
        if (
            epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break
    print(
        "\n" + "=" * 80
    )
    print(
        "TRAINING COMPLETED"
    )
    print(
        "=" * 80
    )
    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )
    print(
        f"Best checkpoint:\n"
        f"{config.BEST_MODEL_PATH}"
    )
    print(
        f"\nTraining history:\n"
        f"{config.TRAINING_HISTORY_PATH}"
    )
if __name__ == "__main__":
    train()