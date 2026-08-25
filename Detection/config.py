from pathlib import Path
import torch
PROJECT_DIR = Path("Paste your path here")
DATASET_DIR = PROJECT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
DATA_YAML = DATASET_DIR / "data.yaml"
TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
VAL_IMAGES_DIR = IMAGES_DIR / "val"
TEST_IMAGES_DIR = IMAGES_DIR / "test"
TRAIN_LABELS_DIR = LABELS_DIR / "train"
VAL_LABELS_DIR = LABELS_DIR / "val"
TEST_LABELS_DIR = LABELS_DIR / "test"
BASE_MODEL = "yolov8n.pt"
BEST_MODEL_PATH = (
    CHECKPOINTS_DIR /
    "best_yolov8_defect_detector.pt"
)
IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 20
WORKERS = 4
NUM_WORKERS = WORKERS
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "yolov8_training"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
NUM_QUALITATIVE_IMAGES = 10
TEST_METRICS_JSON_PATH = (
    METRICS_DIR /
    "test_metrics.json"
)
TEST_METRICS_CSV_PATH = (
    METRICS_DIR /
    "test_overall_metrics.csv"
)
PER_CLASS_METRICS_CSV_PATH = (
    METRICS_DIR /
    "test_per_class_metrics.csv"
)
def create_directories():
    """
    Create all directories required by the project.
    """
    directories = [
        CHECKPOINTS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        PREDICTIONS_DIR,
    ]
    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )