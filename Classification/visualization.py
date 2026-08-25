import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import config
from dataset import (
    create_dataloaders
)
from model import (
    create_model
)
plt.style.use(
    "seaborn-v0_8-whitegrid"
)
sns.set_context(
    "paper"
)
def plot_training_history():
    history = pd.read_csv(
        config.HISTORY_FILE
    )
    plt.figure(
        figsize=config.FIGURE_SIZE
    )
    plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss",
    linewidth=2
    )
    plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss",
    linewidth=2
    )
    plt.xlabel(
    "Epoch"
    )
    plt.ylabel(
    "Loss"
    )
    plt.title(
    "Training and Validation Loss"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "training_validation_loss.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
    plt.figure(
    figsize=config.FIGURE_SIZE
    )
    plt.plot(
    history["epoch"],
    history["train_accuracy"] * 100,
    label="Training Accuracy",
    linewidth=2
    )
    plt.plot(
    history["epoch"],
    history["val_accuracy"] * 100,
    label="Validation Accuracy",
    linewidth=2
    )
    plt.xlabel(
    "Epoch"
    )
    plt.ylabel(
    "Accuracy (%)"
    )
    plt.title(
    "Training and Validation Accuracy"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "training_validation_accuracy.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
    plt.figure(
        figsize=config.FIGURE_SIZE
    )
    plt.plot(
        history["epoch"],
        history["train_f1"],
        label="Training F1",
        linewidth=2
    )
    plt.plot(
        history["epoch"],
        history["val_f1"],
        label="Validation F1",
        linewidth=2
    )
    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "F1-score"
    )
    plt.title(
        "Training and Validation F1-score"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "training_validation_f1.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
    plt.figure(
        figsize=config.FIGURE_SIZE
    )
    plt.plot(
        history["epoch"],
        history["learning_rate"],
        linewidth=2
    )
    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "Learning Rate"
    )
    plt.title(
        "Learning Rate Schedule"
    )
    plt.yscale(
        "log"
    )
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "learning_rate_curve.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
def get_predictions():
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()
    model = create_model()
    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE,
        weights_only=False
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Generating predictions"
        ):
            images = images.to(
                config.DEVICE
            )
            outputs = model(
                images
            )
            probabilities = torch.softmax(
                outputs,
                dim=1
            )
            predictions = torch.argmax(
                probabilities,
                dim=1
            )
            y_true.extend(
                labels.numpy()
            )
            y_pred.extend(
                predictions.cpu().numpy()
            )
            # Probability of crack.
            # crack = class 0
            y_prob.extend(
                probabilities[:, 0]
                .cpu()
                .numpy()
            )
    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
        test_dataset
    )
def plot_confusion_matrix(
    y_true,
    y_pred
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )
    plt.figure(
        figsize=config.FIGURE_SIZE
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar=False
    )
    plt.xlabel(
        "Predicted Class"
    )
    plt.ylabel(
        "True Class"
    )
    plt.title(
        "Confusion Matrix"
    )
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "confusion_matrix.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
def plot_precision_recall(
    y_true,
    y_prob
):
    y_binary = (
        y_true ==
        config.POSITIVE_CLASS_INDEX
    ).astype(int)
    precision, recall, _ = (
        precision_recall_curve(
            y_binary,
            y_prob
        )
    )
    ap = average_precision_score(
        y_binary,
        y_prob
    )
    plt.figure(
        figsize=config.FIGURE_SIZE
    )
    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {ap:.4f}"
    )
    plt.xlabel(
        "Recall"
    )
    plt.ylabel(
        "Precision"
    )
    plt.title(
        "Precision-Recall Curve"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "precision_recall_curve.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
def denormalize(
    image
):
    mean = np.array([
        0.485,
        0.456,
        0.406
    ])
    std = np.array([
        0.229,
        0.224,
        0.225
    ])
    image = image.cpu().numpy()
    image = image.transpose(
        1,
        2,
        0
    )
    image = (
        image * std
        + mean
    )
    image = np.clip(
        image,
        0,
        1
    )
    return image
def plot_sample_predictions():
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()
    model = create_model()
    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE,
        weights_only=False
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    images, labels = next(
        iter(test_loader)
    )
    images_gpu = images.to(
        config.DEVICE
    )
    with torch.no_grad():
        outputs = model(
            images_gpu
        )
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
        predictions = torch.argmax(
            probabilities,
            dim=1
        )
    number_of_images = min(
        8,
        len(images)
    )
    plt.figure(
        figsize=(12, 8)
    )
    for i in range(
        number_of_images
    ):
        plt.subplot(
            2,
            4,
            i + 1
        )
        plt.imshow(
            denormalize(
                images[i]
            )
        )
        plt.axis(
            "off"
        )
        true_class = (
            config.CLASS_NAMES[
                labels[i].item()
            ]
        )
        predicted_class = (
            config.CLASS_NAMES[
                predictions[i].item()
            ]
        )
        confidence = (
            probabilities[
                i,
                predictions[i]
            ].item()
            * 100
        )
        color = (
            "green"
            if labels[i] == predictions[i]
            else "red"
        )
        plt.title(
            f"True: {true_class}\n"
            f"Pred: {predicted_class}\n"
            f"Confidence: {confidence:.1f}%",
            color=color,
            fontsize=9
        )
    plt.tight_layout()
    plt.savefig(
        config.FIGURE_DIR /
        "sample_predictions.png",
        dpi=config.FIGURE_DPI,
        bbox_inches="tight"
    )
    plt.close()
def main():
    print(
        "\nGenerating training curves..."
    )
    plot_training_history()
    print(
        "Generating test predictions..."
    )
    (
        y_true,
        y_pred,
        y_prob,
        dataset
    ) = get_predictions()
    print(
        "Generating confusion matrix..."
    )
    plot_confusion_matrix(
        y_true,
        y_pred
    )
    print(
        "Generating ROC curve..."
    )
    plot_roc_curve(
        y_true,
        y_prob
    )
    print(
        "Generating Precision-Recall curve..."
    )
    plot_precision_recall(
        y_true,
        y_prob
    )
    print(
        "Generating sample predictions..."
    )
    plot_sample_predictions()
    print("\n")
    print("=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)
    print(
        f"Figures saved in:\n"
        f"{config.FIGURE_DIR}"
    )
    print("=" * 70)