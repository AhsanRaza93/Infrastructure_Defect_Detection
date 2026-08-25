import csv
import json
from pathlib import Path
import torch
from tqdm import tqdm
import config
from dataset import get_test_dataset
from model import create_model

def create_confusion_matrix():
    return {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }

def update_confusion_matrix(
    predictions,
    targets,
    confusion,
):
    predictions = predictions.bool()
    targets = targets.bool()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    confusion["TP"] += int(
        (predictions & targets)
        .sum()
        .item()
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
    crack_iou = (
        tp
        / (
            tp + fp + fn + epsilon
        )
    )
    background_iou = (
        tn
        / (
            tn + fp + fn + epsilon
        )
    )
    mean_iou = (
        crack_iou
        + background_iou
    ) / 2.0
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
    f1_score = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + epsilon
        )
    )
    return {
        "iou": float(crack_iou),
        "jaccard_index": float(
            crack_iou
        ),
        "dice": float(dice),
        "pixel_accuracy": float(
            pixel_accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1_score
        ),
        "mean_iou": float(
            mean_iou
        ),
        "background_iou": float(
            background_iou
        ),
        "crack_iou": float(
            crack_iou
        ),
    }

def calculate_single_image_metrics(
    prediction,
    target,
):
    prediction = prediction.bool()
    target = target.bool()
    prediction = prediction.view(-1)
    target = target.view(-1)
    confusion = {
        "TP": int(
            (prediction & target)
            .sum()
            .item()
        ),
        "TN": int(
            ((~prediction) & (~target))
            .sum()
            .item()
        ),
        "FP": int(
            (prediction & (~target))
            .sum()
            .item()
        ),
        "FN": int(
            ((~prediction) & target)
            .sum()
            .item()
        ),
    }
    return calculate_metrics(
        confusion
    )

def load_best_model():
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nBest model checkpoint was not found:\n"
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

def save_json(metrics):
    with open(
        config.TEST_METRICS_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

def save_metrics_csv(
    metrics,
):
    path = (
        config.TEST_METRICS_CSV_PATH
    )
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )
        writer.writerow(
            [
                "Metric",
                "Value",
            ]
        )
        for name, value in metrics.items():
            if isinstance(
                value,
                (int, float),
            ):
                writer.writerow(
                    [
                        name,
                        value,
                    ]
                )

def save_confusion_matrix(
    confusion,
):
    with open(
        config.CONFUSION_MATRIX_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )
        writer.writerow(
            [
                "Quantity",
                "Count",
            ]
        )
        writer.writerow(
            [
                "True Positive",
                confusion["TP"],
            ]
        )
        writer.writerow(
            [
                "True Negative",
                confusion["TN"],
            ]
        )
        writer.writerow(
            [
                "False Positive",
                confusion["FP"],
            ]
        )
        writer.writerow(
            [
                "False Negative",
                confusion["FN"],
            ]
        )

def save_per_image_metrics(
    rows,
):
    if not rows:
        return
    fieldnames = [
        "image",
        "iou",
        "jaccard_index",
        "dice",
        "pixel_accuracy",
        "precision",
        "recall",
        "f1_score",
        "mean_iou",
        "background_iou",
        "crack_iou",
    ]
    with open(
        config.PER_IMAGE_METRICS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

def save_text_report(
    metrics,
    confusion,
    number_of_test_images,
    checkpoint,
):
    with open(
        config.TEST_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "=" * 70 + "\n"
        )
        file.write(
            "CONCRETE CRACK SEGMENTATION "
            "TEST EVALUATION REPORT\n"
        )
        file.write(
            "=" * 70 + "\n\n"
        )
        file.write(
            f"Model: {config.MODEL_NAME}\n"
        )
        file.write(
            f"Encoder: {config.ENCODER_NAME}\n"
        )
        file.write(
            "Encoder weights: ImageNet pretrained\n"
        )
        file.write(
            f"Image size: "
            f"{config.IMAGE_SIZE} x "
            f"{config.IMAGE_SIZE}\n"
        )
        file.write(
            f"Prediction threshold: "
            f"{config.PREDICTION_THRESHOLD}\n"
        )
        file.write(
            f"Test images: "
            f"{number_of_test_images}\n"
        )
        if "epoch" in checkpoint:
            file.write(
                f"Best checkpoint epoch: "
                f"{checkpoint['epoch']}\n"
            )
        if "best_val_loss" in checkpoint:
            file.write(
                f"Best validation loss: "
                f"{checkpoint['best_val_loss']:.6f}\n"
            )
        file.write("\n")
        file.write(
            "-" * 70 + "\n"
        )
        file.write(
            "PIXEL-LEVEL CONFUSION COUNTS\n"
        )
        file.write(
            "-" * 70 + "\n"
        )
        file.write(
            f"True Positive : "
            f"{confusion['TP']:,}\n"
        )
        file.write(
            f"True Negative : "
            f"{confusion['TN']:,}\n"
        )
        file.write(
            f"False Positive: "
            f"{confusion['FP']:,}\n"
        )
        file.write(
            f"False Negative: "
            f"{confusion['FN']:,}\n"
        )
        file.write("\n")
        file.write(
            "-" * 70 + "\n"
        )
        file.write(
            "SEGMENTATION METRICS\n"
        )
        file.write(
            "-" * 70 + "\n"
        )
        metric_order = [
            (
                "IoU / Jaccard Index",
                "iou",
            ),
            (
                "Dice Coefficient",
                "dice",
            ),
            (
                "Pixel Accuracy",
                "pixel_accuracy",
            ),
            (
                "Precision",
                "precision",
            ),
            (
                "Recall",
                "recall",
            ),
            (
                "F1-score",
                "f1_score",
            ),
            (
                "Mean IoU",
                "mean_iou",
            ),
            (
                "Background IoU",
                "background_iou",
            ),
            (
                "Crack IoU",
                "crack_iou",
            ),
        ]
        for display_name, key in metric_order:
            file.write(
                f"{display_name:<25}: "
                f"{metrics[key]:.6f}\n"
            )
        file.write("\n")
        file.write(
            "=" * 70 + "\n"
        )
def print_results(
    metrics,
    confusion,
):
    print("\n" + "=" * 70)
    print(
        "FINAL TEST SET RESULTS"
    )
    print("=" * 70)
    print(
        f"True Positive : "
        f"{confusion['TP']:,}"
    )
    print(
        f"True Negative : "
        f"{confusion['TN']:,}"
    )
    print(
        f"False Positive: "
        f"{confusion['FP']:,}"
    )
    print(
        f"False Negative: "
        f"{confusion['FN']:,}"
    )
    print("-" * 70)
    print(
        f"IoU / Jaccard Index : "
        f"{metrics['iou']:.4f}"
    )
    print(
        f"Dice Coefficient    : "
        f"{metrics['dice']:.4f}"
    )
    print(
        f"Pixel Accuracy      : "
        f"{metrics['pixel_accuracy']:.4f}"
    )
    print(
        f"Precision           : "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall              : "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"F1-score            : "
        f"{metrics['f1_score']:.4f}"
    )
    print(
        f"Mean IoU            : "
        f"{metrics['mean_iou']:.4f}"
    )
    print(
        f"Background IoU      : "
        f"{metrics['background_iou']:.4f}"
    )
    print(
        f"Crack IoU           : "
        f"{metrics['crack_iou']:.4f}"
    )
    print("=" * 70)

@torch.no_grad()
def evaluate():
    print("\n" + "=" * 70)
    print(
        "U-NET + RESNET34 TEST EVALUATION"
    )
    print("=" * 70)
    config.create_directories()
    config.validate_dataset_directories()
    model, checkpoint = (
        load_best_model()
    )
    test_dataset = (
        get_test_dataset()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )
    print(
        f"\nTest images: "
        f"{len(test_dataset):,}"
    )
    confusion = (
        create_confusion_matrix()
    )
    per_image_rows = []
    for (
        images,
        masks,
        image_paths,
    ) in tqdm(
        test_loader,
        desc="Testing",
    ):
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
        probabilities = torch.sigmoid(
            logits
        )
        predictions = (
            probabilities
            >= config.PREDICTION_THRESHOLD
        )
        update_confusion_matrix(
            predictions,
            masks >= 0.5,
            confusion,
        )
        for index in range(
            images.size(0)
        ):
            image_prediction = (
                predictions[index]
            )
            image_target = (
                masks[index] >= 0.5
            )
            image_metrics = (
                calculate_single_image_metrics(
                    image_prediction,
                    image_target,
                )
            )
            row = {
                "image": Path(
                    image_paths[index]
                ).name,
                **image_metrics,
            }
            per_image_rows.append(
                row
            )
    metrics = calculate_metrics(
        confusion
    )
    metrics_with_metadata = {
        "model": config.MODEL_NAME,
        "encoder": config.ENCODER_NAME,
        "image_size": config.IMAGE_SIZE,
        "prediction_threshold": (
            config.PREDICTION_THRESHOLD
        ),
        "test_samples": len(
            test_dataset
        ),
        "checkpoint_epoch": (
            checkpoint.get(
                "epoch",
                None,
            )
        ),
        **metrics,
    }
    print_results(
        metrics,
        confusion,
    )
    save_json(
        metrics_with_metadata
    )
    save_metrics_csv(
        metrics_with_metadata
    )
    save_confusion_matrix(
        confusion
    )
    save_per_image_metrics(
        per_image_rows
    )
    save_text_report(
        metrics=metrics,
        confusion=confusion,
        number_of_test_images=len(
            test_dataset
        ),
        checkpoint=checkpoint,
    )
    print(
        "\nEvaluation completed successfully."
    )
    print(
        f"\nMetrics:\n"
        f"{config.TEST_METRICS_JSON_PATH}"
    )
    print(
        f"Report:\n"
        f"{config.TEST_REPORT_PATH}"
    )
    return (
        metrics,
        confusion,
        per_image_rows,
    )
if __name__ == "__main__":
    evaluate()