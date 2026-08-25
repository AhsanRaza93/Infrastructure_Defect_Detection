from pathlib import Path
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from model import load_model
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
def prepare_directories():
    config.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        config.FIGURES_DIR /
        "qualitative"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        config.FIGURES_DIR /
        "training_curves"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        config.FIGURES_DIR /
        "evaluation"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
def get_training_directory():
    directory = (
        config.RESULTS_DIR /
        config.PROJECT_NAME
    )
    if not directory.exists():
        raise FileNotFoundError(
            "\nTraining directory not found:\n"
            f"{directory}\n\n"
            "Run train.py first."
        )
    return directory
def get_evaluation_directory():
    directory = (
        config.RESULTS_DIR /
        "test_evaluation"
    )
    if not directory.exists():
        raise FileNotFoundError(
            "\nEvaluation directory not found:\n"
            f"{directory}\n\n"
            "Run evaluate.py first."
        )
    return directory
def load_results_csv():
    results_path = (
        get_training_directory() /
        "results.csv"
    )
    if not results_path.exists():
        raise FileNotFoundError(
            "\nresults.csv not found:\n"
            f"{results_path}"
        )
    dataframe = pd.read_csv(
        results_path
    )
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]
    return dataframe
def find_column(
    dataframe,
    candidates,
):
    """
    Find a column while tolerating
    minor naming differences.
    """
    normalized = {
        column.lower().strip(): column
        for column in dataframe.columns
    }
    for candidate in candidates:
        key = (
            candidate.lower().strip()
        )
        if key in normalized:
            return normalized[key]
    return None
def save_figure(filename):
    output_path = (
        config.FIGURES_DIR /
        filename
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved figure: {output_path}"
    )
def plot_training_loss(
    dataframe,
):
    epoch_column = find_column(
        dataframe,
        ["epoch"],
    )
    train_column = find_column(
        dataframe,
        [
            "train/box_loss",
        ],
    )
    val_column = find_column(
        dataframe,
        [
            "val/box_loss",
        ],
    )
    if (
        epoch_column is None
        or train_column is None
        or val_column is None
    ):
        print(
            "WARNING: Box-loss columns "
            "not found."
        )
        return
    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        dataframe[epoch_column],
        dataframe[train_column],
        label="Training Box Loss",
        linewidth=2,
    )
    plt.plot(
        dataframe[epoch_column],
        dataframe[val_column],
        label="Validation Box Loss",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Box Loss")
    plt.title(
        "YOLOv8 Training and Validation Box Loss"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(
        "training_curves/detection_box_loss.png"
    )
def plot_classification_loss(
    dataframe,
):
    epoch_column = find_column(
        dataframe,
        ["epoch"],
    )
    train_column = find_column(
        dataframe,
        ["train/cls_loss"],
    )
    val_column = find_column(
        dataframe,
        ["val/cls_loss"],
    )
    if epoch_column is None:
        return
    if (
        train_column is None
        and val_column is None
    ):
        return
    plt.figure(
        figsize=(8, 5)
    )
    if train_column is not None:
        plt.plot(
            dataframe[epoch_column],
            dataframe[train_column],
            label="Training Classification Loss",
            linewidth=2,
        )
    if val_column is not None:
        plt.plot(
            dataframe[epoch_column],
            dataframe[val_column],
            label="Validation Classification Loss",
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Classification Loss")
    plt.title(
        "YOLOv8 Classification Loss"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(
        "training_curves/detection_classification_loss.png"
    )
def plot_dfl_loss(
    dataframe,
):
    epoch_column = find_column(
        dataframe,
        ["epoch"],
    )
    train_column = find_column(
        dataframe,
        ["train/dfl_loss"],
    )
    val_column = find_column(
        dataframe,
        ["val/dfl_loss"],
    )
    if epoch_column is None:
        return
    if (
        train_column is None
        and val_column is None
    ):
        return
    plt.figure(
        figsize=(8, 5)
    )
    if train_column is not None:
        plt.plot(
            dataframe[epoch_column],
            dataframe[train_column],
            label="Training DFL Loss",
            linewidth=2,
        )
    if val_column is not None:
        plt.plot(
            dataframe[epoch_column],
            dataframe[val_column],
            label="Validation DFL Loss",
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("DFL Loss")
    plt.title(
        "YOLOv8 Distribution Focal Loss"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(
        "training_curves/detection_dfl_loss.png"
    )
def plot_precision(
    dataframe,
):
    epoch_column = find_column(
        dataframe,
        ["epoch"],
    )
    precision_column = find_column(
        dataframe,
        [
            "metrics/precision(B)",
            "metrics/precision",
        ],
    )
    if (
        epoch_column is None
        or precision_column is None
    ):
        print(
            "WARNING: Precision column not found."
        )
        return
    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        dataframe[epoch_column],
        dataframe[precision_column],
        color="tab:blue",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title(
        "YOLOv8 Validation Precision"
    )
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    save_figure(
        "training_curves/detection_precision.png"
    )
def plot_recall(
    dataframe,
):
    epoch_column = find_column(
        dataframe,
        ["epoch"],
    )
    recall_column = find_column(
        dataframe,
        [
            "metrics/recall(B)",
            "metrics/recall",
        ],
    )
    if (
        epoch_column is None
        or recall_column is None
    ):
        print(
            "WARNING: Recall column not found."
        )
        return
    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        dataframe[epoch_column],
        dataframe[recall_column],
        color="tab:green",
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title(
        "YOLOv8 Validation Recall"
    )
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    save_figure(
        "training_curves/detection_recall.png"
    )
def plot_map(
    dataframe,
):
    epoch_column = find_column(
        dataframe,
        ["epoch"],
    )
    map50_column = find_column(
        dataframe,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50",
        ],
    )
    map5095_column = find_column(
        dataframe,
        [
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
        ],
    )
    if epoch_column is None:
        return
    if (
        map50_column is None
        and map5095_column is None
    ):
        print(
            "WARNING: mAP columns not found."
        )
        return
    plt.figure(
        figsize=(8, 5)
    )
    if map50_column is not None:
        plt.plot(
            dataframe[epoch_column],
            dataframe[map50_column],
            label="mAP@0.50",
            linewidth=2,
        )
    if map5095_column is not None:
        plt.plot(
            dataframe[epoch_column],
            dataframe[map5095_column],
            label="mAP@0.50:0.95",
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Average Precision")
    plt.title(
        "YOLOv8 Validation mAP"
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(
        "training_curves/detection_map.png"
    )
def copy_evaluation_figures():
    evaluation_directory = (
        get_evaluation_directory()
    )
    figure_names = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
    ]
    for figure_name in figure_names:
        source = (
            evaluation_directory /
            figure_name
        )
        if not source.exists():
            print(
                f"Evaluation figure not found: "
                f"{source}"
            )
            continue
        destination = (
            config.FIGURES_DIR /
            "evaluation" /
            figure_name
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
            f"Copied evaluation figure: "
            f"{destination}"
        )
def get_test_images():
    image_directory = (
        config.TEST_IMAGES_DIR
    )
    if not image_directory.exists():
        raise FileNotFoundError(
            "\nTest image directory not found:\n"
            f"{image_directory}"
        )
    images = []
    for path in image_directory.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            }
        ):
            images.append(path)
    return sorted(images)
def draw_detections(
    image,
    boxes,
    names,
    color=(0, 165, 255),
    thickness=3,
):
    output = image.copy()
    if boxes is None:
        return output
    if len(boxes) == 0:
        return output
    xyxy = (
        boxes.xyxy
        .cpu()
        .numpy()
    )
    classes = (
        boxes.cls
        .cpu()
        .numpy()
    )
    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )
    height, width = (
        output.shape[:2]
    )
    for (
        box,
        class_id,
        confidence,
    ) in zip(
        xyxy,
        classes,
        confidences,
    ):
        x1, y1, x2, y2 = (
            box.astype(int)
        )
        x1 = max(
            0,
            min(x1, width - 1),
        )
        y1 = max(
            0,
            min(y1, height - 1),
        )
        x2 = max(
            0,
            min(x2, width - 1),
        )
        y2 = max(
            0,
            min(y2, height - 1),
        )
        class_id = int(class_id)
        if isinstance(names, dict):
            class_name = names.get(
                class_id,
                str(class_id),
            )
        else:
            if class_id < len(names):
                class_name = names[
                    class_id
                ]
            else:
                class_name = str(
                    class_id
                )
        label = (
            f"{class_name}: "
            f"{confidence:.2f}"
        )
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )
        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )
        font_scale = 0.55
        text_thickness = 1
        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness,
        )
        text_y = max(
            y1,
            text_height + baseline,
        )
        cv2.rectangle(
            output,
            (
                x1,
                text_y
                - text_height
                - baseline,
            ),
            (
                x1 + text_width,
                text_y,
            ),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (
                x1,
                text_y - baseline,
            ),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return output
def create_prediction_figure(
    model,
    image_path,
    output_path,
):
    image = cv2.imread(
        str(image_path)
    )
    if image is None:
        print(
            f"WARNING: Could not read image: "
            f"{image_path}"
        )
        return
    results = model.predict(
        source=str(image_path),
        imgsz=config.IMAGE_SIZE,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        max_det=config.MAX_DETECTIONS,
        device=config.DEVICE,
        verbose=False,
    )
    if not results:
        return
    result = results[0]
    predicted_image = draw_detections(
        image=image,
        boxes=result.boxes,
        names=model.names,
    )
    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )
    predicted_rgb = cv2.cvtColor(
        predicted_image,
        cv2.COLOR_BGR2RGB,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )
    axes[0].imshow(
        original_rgb
    )
    axes[0].set_title(
        "Original Image"
    )
    axes[0].axis("off")
    axes[1].imshow(
        predicted_rgb
    )
    axes[1].set_title(
        "YOLOv8 Predicted Defects"
    )
    axes[1].axis("off")
    figure.suptitle(
        "Civil Infrastructure Defect Detection",
        fontsize=15,
    )
    plt.tight_layout()
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"Saved qualitative figure: "
        f"{output_path}"
    )
def generate_qualitative_figures(
    model,
    number_of_images=None,
):
    if number_of_images is None:
        number_of_images = (
            config.NUM_QUALITATIVE_IMAGES
        )
    test_images = get_test_images()
    if not test_images:
        raise RuntimeError(
            "No test images were found."
        )
    random_generator = random.Random(
        config.SEED
    )
    number_of_images = min(
        number_of_images,
        len(test_images),
    )
    selected_images = (
        random_generator.sample(
            test_images,
            number_of_images,
        )
    )
    output_directory = (
        config.FIGURES_DIR /
        "qualitative"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    print(
        f"\nGenerating "
        f"{number_of_images} qualitative "
        f"detection figures..."
    )
    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):
        output_path = (
            output_directory /
            f"detection_example_{index:02d}.png"
        )
        create_prediction_figure(
            model=model,
            image_path=image_path,
            output_path=output_path,
        )
def generate_training_figures():
    dataframe = load_results_csv()
    plot_training_loss(
        dataframe
    )
    plot_classification_loss(
        dataframe
    )
    plot_dfl_loss(
        dataframe
    )
    plot_precision(
        dataframe
    )
    plot_recall(
        dataframe
    )
    plot_map(
        dataframe
    )
def main():
    print("\n" + "=" * 80)
    print("YOLOv8 DETECTION VISUALIZATION")
    print("=" * 80)
    config.create_directories()
    prepare_directories()
    print(
        "\nLoading trained model..."
    )
    model = load_model()
    print(
        "\nGenerating training curves..."
    )
    generate_training_figures()
    print(
        "\nOrganizing evaluation figures..."
    )
    try:
        copy_evaluation_figures()
    except FileNotFoundError as error:
        print(
            f"\nWARNING: {error}"
        )
    print(
        "\nGenerating qualitative results..."
    )
    generate_qualitative_figures(
        model=model
    )
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETED")
    print("=" * 80)
if __name__ == "__main__":
    main()