import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import config
from dataset import get_test_dataset
from model import create_model

FIGURE_DPI = 300
FIGURE_SIZE = (
    8,
    6,
)
QUALITATIVE_FIGURE_SIZE = (
    16,
    5,
)
OVERLAY_ALPHA = 0.45

def create_figure_directories():
    config.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    qualitative_dir = (
        config.FIGURES_DIR
        / "qualitative"
    )
    overlays_dir = (
        config.FIGURES_DIR
        / "overlays"
    )
    prediction_dir = (
        config.FIGURES_DIR
        / "predicted_masks"
    )
    qualitative_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    overlays_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return (
        qualitative_dir,
        overlays_dir,
        prediction_dir,
    )

def load_training_history():
    if not config.TRAINING_HISTORY_PATH.exists():
        raise FileNotFoundError(
            "\nTraining history was not found:\n"
            f"{config.TRAINING_HISTORY_PATH}\n\n"
            "Run train.py first."
        )
    with open(
        config.TRAINING_HISTORY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )

def save_figure(
    figure,
    filename,
):
    output_path = (
        config.FIGURES_DIR
        / filename
    )
    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(
        figure
    )
    print(
        f"Saved figure: {output_path}"
    )

def plot_training_curve(
    history,
    train_key,
    val_key,
    title,
    ylabel,
    filename,
    y_min=None,
    y_max=None,
):
    epochs = history["epoch"]
    figure, axis = plt.subplots(
        figsize=FIGURE_SIZE
    )
    axis.plot(
        epochs,
        history[train_key],
        label="Training",
        linewidth=2.2,
        color="#1f77b4",
        marker="o",
        markersize=3,
    )
    axis.plot(
        epochs,
        history[val_key],
        label="Validation",
        linewidth=2.2,
        color="#d62728",
        marker="o",
        markersize=3,
    )
    axis.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel(
        "Epoch",
        fontsize=12,
    )
    axis.set_ylabel(
        ylabel,
        fontsize=12,
    )
    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )
    axis.legend()
    if y_min is not None:
        axis.set_ylim(
            bottom=y_min
        )
    if y_max is not None:
        axis.set_ylim(
            top=y_max
        )
    figure.tight_layout()
    save_figure(
        figure,
        filename,
    )

def plot_loss_curve(history):
    plot_training_curve(
        history,
        "train_loss",
        "val_loss",
        "Training and Validation Loss",
        "Loss",
        "loss_curve.png",
    )

def plot_dice_curve(history):
    plot_training_curve(
        history,
        "train_dice",
        "val_dice",
        "Training and Validation Dice",
        "Dice",
        "dice_curve.png",
        0,
        1,
    )

def plot_iou_curve(history):
    plot_training_curve(
        history,
        "train_iou",
        "val_iou",
        "Training and Validation IoU",
        "IoU / Jaccard",
        "iou_curve.png",
        0,
        1,
    )

def plot_mean_iou_curve(history):
    plot_training_curve(
        history,
        "train_mean_iou",
        "val_mean_iou",
        "Training and Validation Mean IoU",
        "Mean IoU",
        "mean_iou_curve.png",
        0,
        1,
    )

def plot_precision_curve(history):
    plot_training_curve(
        history,
        "train_precision",
        "val_precision",
        "Training and Validation Precision",
        "Precision",
        "precision_curve.png",
        0,
        1,
    )

def plot_recall_curve(history):
    plot_training_curve(
        history,
        "train_recall",
        "val_recall",
        "Training and Validation Recall",
        "Recall",
        "recall_curve.png",
        0,
        1,
    )

def plot_f1_curve(history):
    plot_training_curve(
        history,
        "train_f1_score",
        "val_f1_score",
        "Training and Validation F1-score",
        "F1-score",
        "f1_score_curve.png",
        0,
        1,
    )

def plot_pixel_accuracy_curve(
    history
):
    plot_training_curve(
        history,
        "train_pixel_accuracy",
        "val_pixel_accuracy",
        "Training and Validation Pixel Accuracy",
        "Pixel Accuracy",
        "pixel_accuracy_curve.png",
        0,
        1,
    )

def plot_learning_rate_curve(
    history
):
    epochs = history["epoch"]
    learning_rates = (
        history["learning_rate"]
    )
    figure, axis = plt.subplots(
        figsize=FIGURE_SIZE
    )
    axis.plot(
        epochs,
        learning_rates,
        linewidth=2.2,
        color="#2ca02c",
        marker="o",
        markersize=3,
    )
    axis.set_title(
        "Learning Rate Schedule",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel(
        "Epoch"
    )
    axis.set_ylabel(
        "Learning Rate"
    )
    axis.set_yscale(
        "log"
    )
    axis.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )
    figure.tight_layout()
    save_figure(
        figure,
        "learning_rate_curve.png",
    )

def plot_main_metrics(history):
    epochs = history["epoch"]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )
    metrics = [
        (
            axes[0, 0],
            "train_dice",
            "val_dice",
            "Dice Coefficient",
            "Dice",
        ),
        (
            axes[0, 1],
            "train_iou",
            "val_iou",
            "IoU / Jaccard Index",
            "IoU",
        ),
        (
            axes[1, 0],
            "train_precision",
            "val_precision",
            "Precision",
            "Precision",
        ),
        (
            axes[1, 1],
            "train_recall",
            "val_recall",
            "Recall",
            "Recall",
        ),
    ]
    for (
        axis,
        train_key,
        val_key,
        title,
        ylabel,
    ) in metrics:
        axis.plot(
            epochs,
            history[train_key],
            label="Training",
            color="#1f77b4",
            linewidth=2,
        )
        axis.plot(
            epochs,
            history[val_key],
            label="Validation",
            color="#d62728",
            linewidth=2,
        )
        axis.set_title(
            title,
            fontweight="bold",
        )
        axis.set_xlabel(
            "Epoch"
        )
        axis.set_ylabel(
            ylabel
        )
        axis.set_ylim(
            0,
            1,
        )
        axis.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )
        axis.legend()
    figure.suptitle(
        "U-Net ResNet34 Training Performance",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )
    save_figure(
        figure,
        "main_segmentation_metrics.png",
    )

def plot_final_test_metrics():
    metrics_path = (
        config.TEST_METRICS_JSON_PATH
    )
    if not metrics_path.exists():
        print(
            "\nTest metrics not found."
        )
        print(
            "Run evaluate.py first."
        )
        return
    with open(
        metrics_path,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(
            file
        )
    metric_names = [
        "iou",
        "dice",
        "pixel_accuracy",
        "precision",
        "recall",
        "f1_score",
        "mean_iou",
    ]
    display_names = [
        "IoU",
        "Dice",
        "Pixel\nAccuracy",
        "Precision",
        "Recall",
        "F1",
        "Mean IoU",
    ]
    values = [
        metrics[name]
        for name in metric_names
    ]
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )
    bars = axis.bar(
        display_names,
        values,
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ],
    )
    axis.set_ylim(
        0,
        1.0,
    )
    axis.set_ylabel(
        "Score"
    )
    axis.set_title(
        "Final Test Set Segmentation Performance",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )
    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    figure.tight_layout()
    save_figure(
        figure,
        "final_test_metrics.png",
    )

def load_best_model():
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nBest checkpoint not found:\n"
            f"{config.BEST_MODEL_PATH}\n\n"
            "Run train.py first."
        )
    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE,
    )
    model = create_model(
        pretrained=False
    )
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )
    model = model.to(
        config.DEVICE
    )
    model.eval()
    return (
        model,
        checkpoint,
    )

def denormalize_image(
    image_tensor
):
    mean = torch.tensor(
        config.IMAGENET_MEAN,
        dtype=image_tensor.dtype,
    ).view(
        3,
        1,
        1,
    )
    std = torch.tensor(
        config.IMAGENET_STD,
        dtype=image_tensor.dtype,
    ).view(
        3,
        1,
        1,
    )
    image = (
        image_tensor * std
        + mean
    )
    image = torch.clamp(
        image,
        0,
        1,
    )
    image = (
        image
        .permute(
            1,
            2,
            0,
        )
        .cpu()
        .numpy()
    )
    image = (
        image * 255
    ).astype(
        np.uint8
    )
    return image

def create_overlay(
    image,
    mask,
    alpha=OVERLAY_ALPHA,
):
    image = image.astype(
        np.float32
    )
    mask = np.squeeze(
        mask
    ).astype(bool)
    overlay = image.copy()
    red = np.array(
        [255, 0, 0],
        dtype=np.float32,
    )
    overlay[mask] = (
        (1.0 - alpha)
        * overlay[mask]
        + alpha * red
    )
    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )
def create_boundary_overlay(
    image,
    ground_truth,
    prediction,
):
    image = image.astype(
        np.float32
    )
    gt = np.squeeze(
        ground_truth
    ).astype(bool)
    pred = np.squeeze(
        prediction
    ).astype(bool)
    true_positive = (
        gt & pred
    )
    false_negative = (
        gt & (~pred)
    )
    false_positive = (
        (~gt) & pred
    )
    overlay = image.copy()
    overlay[true_positive] = (
        0.45 * overlay[true_positive]
        + 0.55
        * np.array(
            [255, 255, 0],
            dtype=np.float32,
        )
    )
    overlay[false_negative] = (
        0.45 * overlay[false_negative]
        + 0.55
        * np.array(
            [0, 255, 0],
            dtype=np.float32,
        )
    )
    overlay[false_positive] = (
        0.45 * overlay[false_positive]
        + 0.55
        * np.array(
            [255, 0, 0],
            dtype=np.float32,
        )
    )
    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

def create_qualitative_figure(
    image,
    ground_truth,
    prediction,
    image_name,
    output_path,
):
    figure, axes = plt.subplots(
        1,
        4,
        figsize=QUALITATIVE_FIGURE_SIZE,
    )
    axes[0].imshow(
        image
    )
    axes[0].set_title(
        "Original Image",
        fontweight="bold",
    )
    axes[0].axis("off")
    axes[1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        "Ground Truth Mask",
        fontweight="bold",
    )
    axes[1].axis("off")
    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        "Predicted Mask",
        fontweight="bold",
    )
    axes[2].axis("off")
    overlay = create_overlay(
        image,
        prediction,
    )
    axes[3].imshow(
        overlay
    )
    axes[3].set_title(
        "Segmentation Overlay",
        fontweight="bold",
    )
    axes[3].axis("off")
    figure.suptitle(
        image_name,
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(
        figure
    )

def create_error_analysis_figure(
    image,
    ground_truth,
    prediction,
    image_name,
    output_path,
):
    overlay = create_boundary_overlay(
        image,
        ground_truth,
        prediction,
    )
    figure, axes = plt.subplots(
        1,
        4,
        figsize=QUALITATIVE_FIGURE_SIZE,
    )
    axes[0].imshow(
        image
    )
    axes[0].set_title(
        "Original",
        fontweight="bold",
    )
    axes[0].axis("off")
    axes[1].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        "Ground Truth",
        fontweight="bold",
    )
    axes[1].axis("off")
    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        "Prediction",
        fontweight="bold",
    )
    axes[2].axis("off")
    axes[3].imshow(
        overlay
    )
    axes[3].set_title(
        "Error Analysis",
        fontweight="bold",
    )
    axes[3].axis("off")
    figure.suptitle(
        image_name,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(
        figure
    )

@torch.no_grad()
def generate_qualitative_results(
    number_of_images=None,
):
    (
        qualitative_dir,
        overlays_dir,
        prediction_dir,
    ) = create_figure_directories()
    if number_of_images is None:
        number_of_images = (
            config.NUM_QUALITATIVE_IMAGES
        )
    model, checkpoint = (
        load_best_model()
    )
    test_dataset = (
        get_test_dataset()
    )
    number_of_images = min(
        number_of_images,
        len(test_dataset),
    )
    print(
        f"\nGenerating qualitative "
        f"results for "
        f"{number_of_images} images..."
    )
    for index in tqdm(
        range(number_of_images),
        desc="Generating figures",
    ):
        (
            image_tensor,
            mask_tensor,
            image_path,
        ) = test_dataset[index]
        image_input = (
            image_tensor
            .unsqueeze(0)
            .to(config.DEVICE)
        )
        logits = model(
            image_input
        )
        probabilities = torch.sigmoid(
            logits
        )
        prediction = (
            probabilities
            >= config.PREDICTION_THRESHOLD
        )
        image = denormalize_image(
            image_tensor
        )
        ground_truth = (
            mask_tensor
            .squeeze()
            .cpu()
            .numpy()
        )
        prediction_mask = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        probability_map = (
            probabilities
            .squeeze()
            .cpu()
            .numpy()
        )
        image_name = Path(
            image_path
        ).stem
        comparison_path = (
            qualitative_dir
            / f"{image_name}_comparison.png"
        )
        create_qualitative_figure(
            image=image,
            ground_truth=ground_truth,
            prediction=prediction_mask,
            image_name=Path(
                image_path
            ).name,
            output_path=comparison_path,
        )
        error_path = (
            qualitative_dir
            / f"{image_name}_error_analysis.png"
        )
        create_error_analysis_figure(
            image=image,
            ground_truth=ground_truth,
            prediction=prediction_mask,
            image_name=Path(
                image_path
            ).name,
            output_path=error_path,
        )
        overlay = create_overlay(
            image,
            prediction_mask,
        )
        Image.fromarray(
            overlay
        ).save(
            overlays_dir
            / f"{image_name}_overlay.png"
        )
        prediction_image = (
            prediction_mask * 255
        ).astype(
            np.uint8
        )
        Image.fromarray(
            prediction_image
        ).save(
            prediction_dir
            / f"{image_name}_prediction.png"
        )
        probability_image = (
            probability_map * 255
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )
        Image.fromarray(
            probability_image
        ).save(
            overlays_dir
            / f"{image_name}_probability.png"
        )
    print(
        "\nQualitative visualization completed."
    )

def get_best_epoch(history):
    losses = history[
        "val_loss"
    ]
    index = int(
        np.argmin(losses)
    )
    return history[
        "epoch"
    ][index]

def generate_all_visualizations():
    print(
        "\n" + "=" * 80
    )
    print(
        "SEGMENTATION VISUALIZATION PIPELINE"
    )
    print(
        "=" * 80
    )
    config.create_directories()
    history = (
        load_training_history()
    )
    best_epoch = get_best_epoch(
        history
    )
    print(
        f"\nBest validation-loss epoch: "
        f"{best_epoch}"
    )
    plot_loss_curve(
        history
    )
    plot_dice_curve(
        history
    )
    plot_iou_curve(
        history
    )
    plot_mean_iou_curve(
        history
    )
    plot_precision_curve(
        history
    )
    plot_recall_curve(
        history
    )
    plot_f1_curve(
        history
    )
    plot_pixel_accuracy_curve(
        history
    )
    plot_learning_rate_curve(
        history
    )
    plot_main_metrics(
        history
    )
    plot_final_test_metrics()
    generate_qualitative_results()
    print(
        "\n" + "=" * 80
    )
    print(
        "VISUALIZATION COMPLETE"
    )
    print(
        "=" * 80
    )
    print(
        f"\nFigures saved under:\n"
        f"{config.FIGURES_DIR}"
    )
if __name__ == "__main__":
    generate_all_visualizations()