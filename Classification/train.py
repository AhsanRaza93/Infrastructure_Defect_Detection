import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import config
from dataset import (
    create_dataloaders,
    set_seed
)
from model import (
    create_model
)
if hasattr(torch, "amp"):
    def autocast_context():
        return torch.amp.autocast(
            device_type="cuda",
            enabled=config.USE_AMP
        )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.USE_AMP
    )
else:
    from torch.cuda.amp import autocast, GradScaler
    def autocast_context():
        return autocast(
            enabled=config.USE_AMP
        )
    scaler = GradScaler(
        enabled=config.USE_AMP
    )
def calculate_metrics(
    y_true,
    y_pred
):
    accuracy = accuracy_score(
        y_true,
        y_pred
    )
    precision = precision_score(
        y_true,
        y_pred,
        pos_label=config.POSITIVE_CLASS_INDEX,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        pos_label=config.POSITIVE_CLASS_INDEX,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=config.POSITIVE_CLASS_INDEX,
        zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )
    for images, labels in progress_bar:
        images = images.to(
            config.DEVICE,
            non_blocking=True
        )
        labels = labels.to(
            config.DEVICE,
            non_blocking=True
        )
        optimizer.zero_grad(
            set_to_none=True
        )
        with autocast_context():
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
        scaler.scale(loss).backward()
    if config.GRADIENT_CLIP_NORM is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRADIENT_CLIP_NORM
        )
        scaler.step(
            optimizer
        )
        scaler.update()
        running_loss += loss.item()
        predictions = torch.argmax(
            outputs,
            dim=1
        )
        all_labels.extend(
            labels.detach().cpu().numpy()
        )
        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )
    epoch_loss = (
        running_loss /
        max(len(loader), 1)
    )
    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )
    return epoch_loss, metrics
def validate(
    model,
    loader,
    criterion
):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        progress_bar = tqdm(
            loader,
            desc="Validation",
            leave=False
        )
        for images, labels in progress_bar:
            images = images.to(
                config.DEVICE,
                non_blocking=True
            )
            labels = labels.to(
                config.DEVICE,
                non_blocking=True
            )
            with autocast_context():
                outputs = model(images)
                loss = criterion(
                    outputs,
                    labels
                )
            running_loss += loss.item()
            predictions = torch.argmax(
                outputs,
                dim=1
            )
            all_labels.extend(
                labels.cpu().numpy()
            )
            all_predictions.extend(
                predictions.cpu().numpy()
            )
    epoch_loss = (
        running_loss /
        max(len(loader), 1)
    )
    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )
    return epoch_loss, metrics
def create_optimizer(
    model,
    stage
):
    if stage == 1:
        optimizer = optim.AdamW(
            model.backbone.fc.parameters(),
            lr=config.CLASSIFIER_LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif stage == 2:
        backbone_parameters = []
        classifier_parameters = []
        for name, parameter in model.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("fc."):
                classifier_parameters.append(
                    parameter
                )
            else:
                backbone_parameters.append(
                    parameter
                )
        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": config.BACKBONE_LEARNING_RATE
                },
                {
                    "params": classifier_parameters,
                    "lr": config.CLASSIFIER_LEARNING_RATE
                }
            ],
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(
            "stage must be 1 or 2"
        )
    return optimizer
def create_scheduler(
    optimizer
):
    if not config.USE_SCHEDULER:
        return None
    if config.SCHEDULER_TYPE == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR
        )
    raise ValueError(
        f"Unsupported scheduler: "
        f"{config.SCHEDULER_TYPE}"
    )
def get_learning_rates(
    optimizer
):
    return [
        group["lr"]
        for group in optimizer.param_groups
    ]
def main():
    set_seed()
    print("\n")
    print("=" * 70)
    print("RESNET50 CIVIL INFRASTRUCTURE CRACK CLASSIFICATION")
    print("=" * 70)
    print(
        "\nDevice:",
        config.DEVICE
    )
    print(
        "Total Epochs:",
        config.NUM_EPOCHS
    )
    print(
        "Stage 1 Epochs:",
        config.FREEZE_EPOCHS
    )
    print(
        "Stage 2 Epochs:",
        config.FINE_TUNE_EPOCHS
    )
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()
    print(
        "\nTraining images:",
        len(train_dataset)
    )
    print(
        "Validation images:",
        len(val_dataset)
    )
    print(
        "Testing images:",
        len(test_dataset)
    )
    model = create_model()
    print(
        "\nModel created successfully."
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.LABEL_SMOOTHING
    )
    print("\n")
    print("=" * 70)
    print("STAGE 1: TRANSFER LEARNING")
    print("Classification head only")
    print("=" * 70)
    model.freeze_backbone()
    model.set_head_training_mode()
    optimizer = create_optimizer(
        model,
        stage=1
    )
    scheduler = create_scheduler(
        optimizer
    )
    best_f1 = -1.0
    patience_counter = 0
    history = []
    global_epoch = 0
    for stage_epoch in range(
        config.FREEZE_EPOCHS
    ):
        global_epoch += 1
        start_time = time.time()
        print("\n")
        print(
            f"Stage 1 | Epoch "
            f"{stage_epoch + 1}/"
            f"{config.FREEZE_EPOCHS}"
        )
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion
        )
        if scheduler is not None:
            scheduler.step(
                val_metrics["f1"]
            )
        learning_rates = get_learning_rates(
            optimizer
        )
        epoch_time = (
            time.time() -
            start_time
        )
        history.append({
            "epoch": global_epoch,
            "stage": "transfer_learning",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "learning_rate": learning_rates[0],
            "epoch_time_seconds": epoch_time
        })
        print(
            f"Train Loss      : {train_loss:.4f}"
        )
        print(
            f"Val Loss        : {val_loss:.4f}"
        )
        print(
            f"Train Accuracy  : "
            f"{train_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"Val Accuracy    : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"Val Precision   : "
            f"{val_metrics['precision']:.4f}"
        )
        print(
            f"Val Recall      : "
            f"{val_metrics['recall']:.4f}"
        )
        print(
            f"Val F1          : "
            f"{val_metrics['f1']:.4f}"
        )
        print(
            f"Learning Rate   : "
            f"{learning_rates[0]:.2e}"
        )
        print(
            f"Time            : "
            f"{epoch_time:.2f}s"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            checkpoint = {
                "epoch": global_epoch,
                "stage": 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict()
                    if scheduler is not None
                    else None,
                "best_val_f1":
                    best_f1,
                "class_names":
                    config.CLASS_NAMES
            }
            torch.save(
                checkpoint,
                config.BEST_MODEL_PATH
            )
            print(
                "Best model saved."
            )
        else:
            patience_counter += 1
    print("\n")
    print("=" * 70)
    print("STAGE 2: FULL FINE-TUNING")
    print("Entire ResNet50 network")
    print("=" * 70)
    model.unfreeze_backbone()
    model.train()
    optimizer = create_optimizer(
        model,
        stage=2
    )
    scheduler = create_scheduler(
        optimizer
    )
    patience_counter = 0
    for stage_epoch in range(
        config.FINE_TUNE_EPOCHS
    ):
        global_epoch += 1
        start_time = time.time()
        print("\n")
        print(
            f"Stage 2 | Epoch "
            f"{stage_epoch + 1}/"
            f"{config.FINE_TUNE_EPOCHS}"
        )
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion
        )
        if scheduler is not None:
            scheduler.step(
                val_metrics["f1"]
            )
        learning_rates = get_learning_rates(
            optimizer
        )
        epoch_time = (
            time.time() -
            start_time
        )
        history.append({
            "epoch": global_epoch,
            "stage": "fine_tuning",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "learning_rate":
                learning_rates[-1],
            "epoch_time_seconds":
                epoch_time
        })
        print(
            f"Train Loss      : {train_loss:.4f}"
        )
        print(
            f"Val Loss        : {val_loss:.4f}"
        )
        print(
            f"Train Accuracy  : "
            f"{train_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"Val Accuracy    : "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"Val Precision   : "
            f"{val_metrics['precision']:.4f}"
        )
        print(
            f"Val Recall      : "
            f"{val_metrics['recall']:.4f}"
        )
        print(
            f"Val F1          : "
            f"{val_metrics['f1']:.4f}"
        )
        print(
            f"Backbone LR     : "
            f"{learning_rates[0]:.2e}"
        )
        print(
            f"Classifier LR   : "
            f"{learning_rates[-1]:.2e}"
        )
        print(
            f"Time            : "
            f"{epoch_time:.2f}s"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 
            checkpoint = {
                "epoch": global_epoch,
                "stage": 2,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict()
                    if scheduler is not None
                    else None,
                "best_val_f1":
                    best_f1,
                "class_names":
                    config.CLASS_NAMES
            }
            torch.save(
                checkpoint,
                config.BEST_MODEL_PATH
            )
            print(
                "Best model saved."
            )
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: "
                f"{patience_counter}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )
        last_checkpoint = {
            "epoch": global_epoch,
            "stage": 2,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict()
                if scheduler is not None
                else None,
            "best_val_f1":
                best_f1
        }
        torch.save(
            last_checkpoint,
            config.LAST_MODEL_PATH
        )
        if (
            config.USE_EARLY_STOPPING
            and
            patience_counter >=
            config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break
    history_df = pd.DataFrame(
        history
    )
    history_df.to_csv(
        config.HISTORY_FILE,
        index=False
    )
    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(
        f"Best Validation F1: "
        f"{best_f1:.4f}"
    )
    print(
        f"Best model: "
        f"{config.BEST_MODEL_PATH}"
    )
    print(
        f"Training history: "
        f"{config.HISTORY_FILE}"
    )
    print("=" * 70)
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
if __name__ == "__main__":
    main()