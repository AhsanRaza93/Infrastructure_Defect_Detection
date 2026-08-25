from pathlib import Path
import csv
import json
import math
import shutil
import numpy as np
from ultralytics import YOLO
import config
from dataset import validate_dataset
from model import load_model
def safe_float(value):
    """
    Convert a value to a JSON/CSV-safe float.
    """
    try:
        value = float(value)
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return 0.0
        return value
    except (
        TypeError,
        ValueError,
    ):
        return 0.0
def prepare_output_directories():
    config.METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    config.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
def run_test_evaluation(model):
    """
    Evaluate the trained model on the independent test split.
    """
    print("\n" + "=" * 80)
    print("YOLOv8 TEST-SET EVALUATION")
    print("=" * 80)
    print(
        f"Dataset  : {config.DATA_YAML}"
    )
    print(
        f"Image size: {config.IMAGE_SIZE}"
    )
    print("=" * 80)
    metrics = model.val(
        data=str(config.DATA_YAML),
        split="test",
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        max_det=config.MAX_DETECTIONS,
        workers=config.NUM_WORKERS,
        plots=True,
        save_json=True,
        project=str(config.RESULTS_DIR),
        name="test_evaluation",
        exist_ok=True,
        verbose=True,
    )
    return metrics
def extract_overall_metrics(metrics):
    box_metrics = metrics.box
    precision = safe_float(
        box_metrics.mp
    )
    recall = safe_float(
        box_metrics.mr
    )
    map50 = safe_float(
        box_metrics.map50
    )
    map5095 = safe_float(
        box_metrics.map
    )
    if precision + recall > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": safe_float(f1),
        "mAP@0.50": map50,
        "mAP@0.50:0.95": map5095,
    }
def extract_per_class_metrics(
    metrics,
    class_names,
):
    """
    Extract per-class metrics from Ultralytics.
    Ultralytics provides:
        p
        r
        ap50
        ap
    ap is normally an array with:
        [number_of_classes, number_of_iou_thresholds]
    The mean over IoU thresholds gives AP@0.50:0.95.
    """
    box_metrics = metrics.box
    precision_values = np.asarray(
        getattr(
            box_metrics,
            "p",
            [],
        ),
        dtype=float,
    ).reshape(-1)
    recall_values = np.asarray(
        getattr(
            box_metrics,
            "r",
            [],
        ),
        dtype=float,
    ).reshape(-1)
    ap50_values = np.asarray(
        getattr(
            box_metrics,
            "ap50",
            [],
        ),
        dtype=float,
    ).reshape(-1)
    ap_values = np.asarray(
        getattr(
            box_metrics,
            "ap",
            [],
        ),
        dtype=float,
    )
    number_of_classes = len(
        class_names
    )
    per_class = []
    for class_id in range(
        number_of_classes
    ):

        if class_id < len(
            precision_values
        ):

            precision = safe_float(
                precision_values[class_id]
            )

        else:
            precision = 0.0

        if class_id < len(
            recall_values
        ):

            recall = safe_float(
                recall_values[class_id]
            )

        else:
            recall = 0.0

        if class_id < len(
            ap50_values
        ):

            ap50 = safe_float(
                ap50_values[class_id]
            )

        else:
            ap50 = 0.0

        ap5095 = 0.0

        if ap_values.ndim == 2:

            if (
                class_id <
                ap_values.shape[0]
            ):

                class_ap = ap_values[
                    class_id
                ]

                if class_ap.size > 0:

                    ap5095 = safe_float(
                        np.mean(class_ap)
                    )

        elif ap_values.ndim == 1:

            if class_id < len(
                ap_values
            ):

                ap5095 = safe_float(
                    ap_values[class_id]
                )

        if precision + recall > 0:

            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )

        else:
            f1 = 0.0

        per_class.append(
            {
                "class_id": class_id,

                "class_name": class_names[
                    class_id
                ],

                "precision": safe_float(
                    precision
                ),

                "recall": safe_float(
                    recall
                ),

                "f1_score": safe_float(
                    f1
                ),

                "AP@0.50": safe_float(
                    ap50
                ),

                "AP@0.50:0.95": safe_float(
                    ap5095
                ),
            }
        )

    return per_class
def save_json_report(
    overall_metrics,
    per_class_metrics,
):
    report = {
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
    }
    with open(
        config.TEST_METRICS_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )
    print(
        "\nJSON report saved:"
    )
    print(
        config.TEST_METRICS_JSON_PATH
    )

def save_overall_csv(
    overall_metrics,
):
    fieldnames = [
        "precision",
        "recall",
        "f1_score",
        "mAP@0.50",
        "mAP@0.50:0.95",
    ]
    with open(
        config.TEST_METRICS_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerow(
            overall_metrics
        )
    print(
        "\nOverall CSV saved:"
    )
    print(
        config.TEST_METRICS_CSV_PATH
    )

def save_per_class_csv(
    per_class_metrics,
):
    fieldnames = [
        "class_id",
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "AP@0.50",
        "AP@0.50:0.95",
    ]
    with open(
        config.PER_CLASS_METRICS_CSV_PATH,
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
            per_class_metrics
        )
    print(
        "\nPer-class CSV saved:"
    )
    print(
        config.PER_CLASS_METRICS_CSV_PATH
    )

def print_metrics(
    overall_metrics,
    per_class_metrics,
):
    print("\n" + "=" * 80)
    print("OVERALL TEST METRICS")
    print("=" * 80)
    print(
        f"Precision        : "
        f"{overall_metrics['precision']:.4f}"
    )
    print(
        f"Recall           : "
        f"{overall_metrics['recall']:.4f}"
    )
    print(
        f"F1-score         : "
        f"{overall_metrics['f1_score']:.4f}"
    )
    print(
        f"mAP@0.50         : "
        f"{overall_metrics['mAP@0.50']:.4f}"
    )
    print(
        f"mAP@0.50:0.95    : "
        f"{overall_metrics['mAP@0.50:0.95']:.4f}"
    )
    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)
    for item in per_class_metrics:
        print(
            f"\nClass {item['class_id']}: "
            f"{item['class_name']}"
        )
        print(
            f"  Precision     : "
            f"{item['precision']:.4f}"
        )
        print(
            f"  Recall        : "
            f"{item['recall']:.4f}"
        )
        print(
            f"  F1-score      : "
            f"{item['f1_score']:.4f}"
        )
        print(
            f"  AP@0.50       : "
            f"{item['AP@0.50']:.4f}"
        )
        print(
            f"  AP@0.50:0.95  : "
            f"{item['AP@0.50:0.95']:.4f}"
        )
    print("\n" + "=" * 80)

def copy_evaluation_figures():
    evaluation_directory = (
        config.RESULTS_DIR /
        "test_evaluation"
    )
    if not evaluation_directory.exists():
        print(
            "\nWARNING: Evaluation directory "
            "was not found:"
        )
        print(
            evaluation_directory
        )
        return
    figure_names = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    for figure_name in figure_names:
        source = (
            evaluation_directory /
            figure_name
        )
        if not source.exists():
            continue
        destination = (
            config.FIGURES_DIR /
            f"detection_{figure_name}"
        )
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(
            source,
            destination,
        )
        print(
            f"Copied: {figure_name}"
        )
def main():
    print("\n" + "=" * 80)
    print("CIVIL INFRASTRUCTURE DEFECT DETECTION")
    print("YOLOv8 TEST EVALUATION")
    print("=" * 80)
    config.create_directories()
    prepare_output_directories()
    print(
        "\nStep 1: Validating dataset..."
    )
    dataset_info = validate_dataset()
    print(
        "\nStep 2: Loading best trained model..."
    )
    model = load_model()
    print(
        "\nStep 3: Evaluating independent test set..."
    )
    metrics = run_test_evaluation(
        model
    )
    print(
        "\nStep 4: Extracting metrics..."
    )
    overall_metrics = (
        extract_overall_metrics(
            metrics
        )
    )
    per_class_metrics = (
        extract_per_class_metrics(
            metrics=metrics,
            class_names=dataset_info["names"],
        )
    )
    print(
        "\nStep 5: Saving metric reports..."
    )
    save_json_report(
        overall_metrics,
        per_class_metrics,
    )
    save_overall_csv(
        overall_metrics
    )
    save_per_class_csv(
        per_class_metrics
    )
    print_metrics(
        overall_metrics,
        per_class_metrics,
    )
    print(
        "\nStep 6: Organizing evaluation figures..."
    )
    copy_evaluation_figures()
    print("\n" + "=" * 80)
    print("TEST EVALUATION COMPLETED")
    print("=" * 80)
if __name__ == "__main__":
    main()