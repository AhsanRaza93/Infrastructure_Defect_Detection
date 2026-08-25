from pathlib import Path
import csv
import json
import cv2
import config
from model import load_model
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
def prepare_prediction_directories():
    config.PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        config.PREDICTIONS_DIR /
        "images"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        config.PREDICTIONS_DIR /
        "labels"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        config.PREDICTIONS_DIR /
        "reports"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
def find_images(
    input_path,
):
    input_path = Path(
        input_path
    )
    if not input_path.exists():
        raise FileNotFoundError(
            f"\nInput path does not exist:\n"
            f"{input_path}"
        )
    if input_path.is_file():
        if (
            input_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            raise ValueError(
                f"\nUnsupported image format:\n"
                f"{input_path.suffix}"
            )
        return [input_path]
    images = []
    for path in input_path.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            images.append(path)
    return sorted(images)
def predict_image(
    model,
    image_path,
):
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
        return None
    return results[0]
def extract_predictions(
    result,
):
    predictions = []
    if result is None:
        return predictions
    boxes = result.boxes
    if boxes is None:
        return predictions
    if len(boxes) == 0:
        return predictions
    xyxy = (
        boxes.xyxy
        .cpu()
        .numpy()
    )
    xywhn = (
        boxes.xywhn
        .cpu()
        .numpy()
    )
    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )
    class_ids = (
        boxes.cls
        .cpu()
        .numpy()
    )
    names = result.names
    for index in range(
        len(boxes)
    ):
        class_id = int(
            class_ids[index]
        )
        confidence = float(
            confidences[index]
        )
        x1, y1, x2, y2 = (
            xyxy[index]
        )
        (
            x_center,
            y_center,
            width,
            height,
        ) = xywhn[index]
        if isinstance(
            names,
            dict,
        ):
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
        predictions.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "x_center": float(
                    x_center
                ),
                "y_center": float(
                    y_center
                ),
                "width": float(width),
                "height": float(height),
            }
        )
    return predictions
def save_yolo_labels(
    predictions,
    output_path,
):
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions:
            file.write(
                f"{prediction['class_id']} "
                f"{prediction['x_center']:.6f} "
                f"{prediction['y_center']:.6f} "
                f"{prediction['width']:.6f} "
                f"{prediction['height']:.6f}\n"
            )
def draw_predictions(
    image,
    predictions,
):
    output = image.copy()
    for prediction in predictions:
        x1 = int(
            prediction["x1"]
        )
        y1 = int(
            prediction["y1"]
        )
        x2 = int(
            prediction["x2"]
        )
        y2 = int(
            prediction["y2"]
        )
        class_name = (
            prediction["class_name"]
        )
        confidence = (
            prediction["confidence"]
        )
        color = (
            0,
            165,
            255,
        )
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )
        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )
        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )
        font_scale = 0.6
        thickness = 2
        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
        label_y = max(
            y1,
            text_height + baseline,
        )
        cv2.rectangle(
            output,
            (
                x1,
                label_y
                - text_height
                - baseline,
            ),
            (
                x1 + text_width,
                label_y,
            ),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (
                x1,
                label_y - baseline,
            ),
            font,
            font_scale,
            (
                255,
                255,
                255,
            ),
            thickness,
            cv2.LINE_AA,
        )
    return output
def save_annotated_image(
    image,
    output_path,
):
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    success = cv2.imwrite(
        str(output_path),
        image,
    )
    if not success:
        raise IOError(
            "\nCould not save image:\n"
            f"{output_path}"
        )
def save_prediction_json(
    image_path,
    predictions,
    output_path,
):
    report = {
        "image": str(image_path),
        "number_of_detections": len(
            predictions
        ),
        "detections": predictions,
    }
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )
def save_prediction_csv(
    rows,
    output_path,
):
    fieldnames = [
        "image",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "x_center",
        "y_center",
        "width",
        "height",
    ]
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        output_path,
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
def predict_single_image(
    model,
    image_path,
):
    image_path = Path(
        image_path
    )
    image = cv2.imread(
        str(image_path)
    )
    if image is None:
        raise ValueError(
            f"\nCould not read image:\n"
            f"{image_path}"
        )
    result = predict_image(
        model=model,
        image_path=image_path,
    )
    predictions = extract_predictions(
        result
    )
    annotated_image = draw_predictions(
        image=image,
        predictions=predictions,
    )
    stem = image_path.stem
    image_output = (
        config.PREDICTIONS_DIR /
        "images" /
        f"{stem}_prediction.jpg"
    )
    label_output = (
        config.PREDICTIONS_DIR /
        "labels" /
        f"{stem}.txt"
    )
    json_output = (
        config.PREDICTIONS_DIR /
        "reports" /
        f"{stem}.json"
    )
    save_annotated_image(
        annotated_image,
        image_output,
    )
    save_yolo_labels(
        predictions,
        label_output,
    )
    save_prediction_json(
        image_path,
        predictions,
        json_output,
    )
    print(
        f"\nDetections: "
        f"{len(predictions)}"
    )
    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        print(
            f"\nDetection {index}:"
        )
        print(
            f"  Class      : "
            f"{prediction['class_name']}"
        )
        print(
            f"  Confidence : "
            f"{prediction['confidence']:.4f}"
        )
        print(
            f"  Bounding box: "
            f"("
            f"{prediction['x1']:.1f}, "
            f"{prediction['y1']:.1f}, "
            f"{prediction['x2']:.1f}, "
            f"{prediction['y2']:.1f}"
            f")"
        )
    print(
        "\nSaved prediction image:"
    )
    print(image_output)
    return predictions
def predict_directory(
    model,
    input_directory,
):
    input_directory = Path(
        input_directory
    )
    images = find_images(
        input_directory
    )
    if not images:
        raise RuntimeError(
            f"\nNo images found in:\n"
            f"{input_directory}"
        )
    print("\n" + "=" * 70)
    print("YOLOv8 DIRECTORY PREDICTION")
    print("=" * 70)
    print(
        f"Input directory: "
        f"{input_directory}"
    )
    print(
        f"Number of images: "
        f"{len(images)}"
    )
    all_rows = []
    for index, image_path in enumerate(
        images,
        start=1,
    ):
        print(
            f"\n[{index}/{len(images)}] "
            f"{image_path.name}"
        )
        image = cv2.imread(
            str(image_path)
        )
        if image is None:
            print(
                "WARNING: Image could not "
                "be read. Skipping."
            )
            continue
        result = predict_image(
            model=model,
            image_path=image_path,
        )
        predictions = extract_predictions(
            result
        )
        annotated_image = draw_predictions(
            image=image,
            predictions=predictions,
        )
        try:
            relative_path = (
                image_path.relative_to(
                    input_directory
                )
            )
        except ValueError:
            relative_path = (
                Path(image_path.name)
            )
        relative_path = Path(
            relative_path
        )
        output_stem = (
            relative_path.stem
        )
        image_output = (
            config.PREDICTIONS_DIR /
            "images" /
            relative_path.parent /
            f"{output_stem}_prediction.jpg"
        )
        label_output = (
            config.PREDICTIONS_DIR /
            "labels" /
            relative_path.parent /
            f"{output_stem}.txt"
        )
        json_output = (
            config.PREDICTIONS_DIR /
            "reports" /
            relative_path.parent /
            f"{output_stem}.json"
        )
        save_annotated_image(
            annotated_image,
            image_output,
        )
        save_yolo_labels(
            predictions,
            label_output,
        )
        save_prediction_json(
            image_path,
            predictions,
            json_output,
        )
        for prediction in predictions:
            all_rows.append(
                {
                    "image": str(
                        relative_path
                    ),
                    "class_id": prediction[
                        "class_id"
                    ],
                    "class_name": prediction[
                        "class_name"
                    ],
                    "confidence": prediction[
                        "confidence"
                    ],
                    "x1": prediction["x1"],
                    "y1": prediction["y1"],
                    "x2": prediction["x2"],
                    "y2": prediction["y2"],
                    "x_center": prediction[
                        "x_center"
                    ],
                    "y_center": prediction[
                        "y_center"
                    ],
                    "width": prediction[
                        "width"
                    ],
                    "height": prediction[
                        "height"
                    ],
                }
            )
        print(
            f"  Detections: "
            f"{len(predictions)}"
        )
    csv_output = (
        config.PREDICTIONS_DIR /
        "reports" /
        "all_predictions.csv"
    )
    save_prediction_csv(
        all_rows,
        csv_output,
    )
    print("\n" + "=" * 70)
    print("DIRECTORY PREDICTION COMPLETED")
    print("=" * 70)
    print(
        f"\nPrediction images:"
    )
    print(
        config.PREDICTIONS_DIR /
        "images"
    )
    print(
        f"\nPrediction labels:"
    )
    print(
        config.PREDICTIONS_DIR /
        "labels"
    )
    print(
        f"\nPrediction reports:"
    )
    print(
        config.PREDICTIONS_DIR /
        "reports"
    )
    return all_rows
def predict_test_set(
    model,
):
    return predict_directory(
        model=model,
        input_directory=config.TEST_IMAGES_DIR,
    )
def main():
    print("\n" + "=" * 80)
    print("CIVIL INFRASTRUCTURE DEFECT DETECTION")
    print("YOLOv8 PREDICTION")
    print("=" * 80)
    config.create_directories()
    prepare_prediction_directories()
    print(
        "\nLoading best trained YOLOv8 model..."
    )
    model = load_model()
    print(
        "\nRunning prediction on test images..."
    )
    predict_test_set(
        model
    )
    print(
        "\nPrediction process completed."
    )
if __name__ == "__main__":
    main()