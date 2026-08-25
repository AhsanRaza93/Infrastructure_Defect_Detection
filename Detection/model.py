from pathlib import Path
from ultralytics import YOLO
import config
def load_base_model():
    """
    Load the pretrained YOLOv8 model used for training.
    """
    print(
        f"\nLoading base model: "
        f"{config.BASE_MODEL}"
    )
    return YOLO(config.BASE_MODEL)
def load_model(
    model_path=None,
):
    """
    Load the trained YOLOv8 checkpoint.
    If model_path is not supplied, the function uses:
    detection/checkpoints/
    best_yolov8_defect_detector.pt
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "\nTrained YOLOv8 checkpoint not found:\n"
            f"{model_path}\n\n"
            "Run train.py first."
        )
    print(
        f"Loading trained model:\n"
        f"{model_path}"
    )
    return YOLO(str(model_path))