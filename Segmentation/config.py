from pathlib import Path
import torch
import matplotlib.pyplot as plt

ROOT_DIR = Path(
    "Paste your path here"
)

SEGMENTATION_DIR = Path(__file__).resolve().parent

DATASET_DIR = SEGMENTATION_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

TRAIN_IMAGE_DIR = TRAIN_DIR / "images"
TRAIN_MASK_DIR = TRAIN_DIR / "masks"

VAL_IMAGE_DIR = VAL_DIR / "images"
VAL_MASK_DIR = VAL_DIR / "masks"

TEST_IMAGE_DIR = TEST_DIR / "images"
TEST_MASK_DIR = TEST_DIR / "masks"

CHECKPOINTS_DIR = SEGMENTATION_DIR / "checkpoints"
FIGURES_DIR = SEGMENTATION_DIR / "figures"
RESULTS_DIR = SEGMENTATION_DIR / "results"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"

METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_dataset_directories():

    required_directories = {
        "Training images": TRAIN_IMAGE_DIR,
        "Training masks": TRAIN_MASK_DIR,
        "Validation images": VAL_IMAGE_DIR,
        "Validation masks": VAL_MASK_DIR,
        "Test images": TEST_IMAGE_DIR,
        "Test masks": TEST_MASK_DIR,
    }

    print("\nChecking dataset directories...")
    print("=" * 60)

    all_valid = True

    for name, directory in required_directories.items():

        if directory.exists():
            print(f"✓ {name:<22}: {directory}")

        else:
            print(f"✗ {name:<22}: NOT FOUND")
            print(f"  Expected: {directory}")
            all_valid = False

    print("=" * 60)

    if not all_valid:
        raise FileNotFoundError(
            "One or more required dataset directories are missing."
        )

    print("✓ All dataset directories are valid.")

def create_directories():
    """Create all required project directories if they don't exist."""

    directories = [
        CHECKPOINTS_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        PREDICTIONS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("✓ Project directories are ready.")

def print_configuration():
    """Print the current project configuration."""

    print("\n")
    print("=" * 80)
    print("PROJECT CONFIGURATION")
    print("=" * 80)

    print("\nDIRECTORIES")
    print("-" * 80)
    print(f"Segmentation : {SEGMENTATION_DIR}")
    print(f"Dataset      : {DATASET_DIR}")
    print(f"Train        : {TRAIN_DIR}")
    print(f"Validation   : {VAL_DIR}")
    print(f"Test         : {TEST_DIR}")

    print("\nIMAGE / MASK DIRECTORIES")
    print("-" * 80)
    print(f"Train images : {TRAIN_IMAGE_DIR}")
    print(f"Train masks  : {TRAIN_MASK_DIR}")
    print(f"Val images   : {VAL_IMAGE_DIR}")
    print(f"Val masks    : {VAL_MASK_DIR}")
    print(f"Test images  : {TEST_IMAGE_DIR}")
    print(f"Test masks   : {TEST_MASK_DIR}")

    print("\nOUTPUT DIRECTORIES")
    print("-" * 80)
    print(f"Checkpoints  : {CHECKPOINTS_DIR}")
    print(f"Figures      : {FIGURES_DIR}")
    print(f"Results      : {RESULTS_DIR}")
    print(f"Metrics      : {METRICS_DIR}")
    print(f"Reports      : {REPORTS_DIR}")
    print(f"Predictions  : {PREDICTIONS_DIR}")

    print("\nMODEL CONFIGURATION")
    print("-" * 80)
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Num classes  : {NUM_CLASSES}")
    print(f"Input channels: {IN_CHANNELS}")

    print("\nDEVICE")
    print("-" * 80)
    print(f"Device       : {DEVICE}")

    print("=" * 80)

IMAGE_SIZE = 256

NUM_CLASSES = 1

IN_CHANNELS = 3

PRETRAINED_ENCODER = True

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

RANDOM_SEED = 42

BATCH_SIZE = 8

NUM_WORKERS = 0

PIN_MEMORY = True

NUM_EPOCHS = 50

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MODEL_NAME = "UNetResNet34"

ENCODER_NAME = "ResNet34"

PREDICTION_THRESHOLD = 0.5

EARLY_STOPPING_PATIENCE = 10

NUM_QUALITATIVE_IMAGES = 10

USE_SCHEDULER = True

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 3

MIN_LEARNING_RATE = 1e-7

SAVE_BEST_MODEL = True

SAVE_FINAL_MODEL = True

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"

LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"

HISTORY_FILE = ROOT_DIR / "training_history.csv"

TEST_METRICS_CSV_PATH = ROOT_DIR / "test_metrics.csv"

TEXT_REPORT_PATH = (
    ROOT_DIR / "test_report.csv"
)

PREDICTIONS_FILE = (
    ROOT_DIR / "test_predictions.csv"
)

FINAL_MODEL_PATH = CHECKPOINTS_DIR / "final_model.pth"

TEST_METRICS_JSON_PATH = ROOT_DIR / "test_metrics.json"

CONFUSION_MATRIX_CSV_PATH = ROOT_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PATH = ROOT_DIR / "confusion_matrix.png"

PER_IMAGE_METRICS_CSV_PATH = ROOT_DIR / "per_image_metrics.csv"
PER_IMAGE_METRICS_PATH = ROOT_DIR / "per_image_metrics.png"