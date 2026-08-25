from pathlib import Path
import torch

ROOT_DIR = Path(
    "Paste your path here"
)
DATA_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

print("Root:", ROOT_DIR)
print("Dataset:", DATA_DIR)
print("Train:", TRAIN_DIR)
print("Validation:", VAL_DIR)
print("Test:", TEST_DIR)

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULT_DIR = ROOT_DIR / "results"
FIGURE_DIR = ROOT_DIR / "figures"
LOG_DIR = ROOT_DIR / "logs"
for folder in [
    CHECKPOINT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    LOG_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

EXPERIMENT_NAME = "resnet50_civil_infrastructure_crack_classification"
DATASET_NAME = "YOUR_DATASET_NAME"
DATASET_VERSION = "v1"

IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = [
    "crack",
    "no-crack"
]
POSITIVE_CLASS = "crack"
NEGATIVE_CLASS = "no-crack"
POSITIVE_CLASS_INDEX = 0
NEGATIVE_CLASS_INDEX = 1

BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

MODEL_NAME = "ResNet50"
PRETRAINED = True

FREEZE_BACKBONE = True
FREEZE_EPOCHS = 5
FINE_TUNE_EPOCHS = 25
NUM_EPOCHS = FREEZE_EPOCHS + FINE_TUNE_EPOCHS

BACKBONE_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.0
GRADIENT_CLIP_NORM = None
MODEL_SELECTION_METRIC = "f1"
USE_SCHEDULER = True
SCHEDULER_TYPE = "ReduceLROnPlateau"
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 3
MIN_LR = 1e-7
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7
SEED = 42
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
USE_AMP = torch.cuda.is_available()
BEST_MODEL_PATH = (
    CHECKPOINT_DIR /
    "best_resnet50_crack_classifier.pth"
)
LAST_MODEL_PATH = (
    CHECKPOINT_DIR /
    "last_resnet50_crack_classifier.pth"
)
HISTORY_FILE = (
    RESULT_DIR /
    "training_history.csv"
)
CLASSIFICATION_REPORT_FILE = (
    RESULT_DIR /
    "classification_report.csv"
)
METRICS_FILE = (
    RESULT_DIR /
    "test_metrics.csv"
)
PREDICTIONS_FILE = (
    RESULT_DIR /
    "test_predictions.csv"
)
FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
PRINT_INTERVAL = 1
SAVE_HISTORY = True
DEBUG = False

import torch
import torch.nn as nn

print("\n" + "=" * 70)
print("RESNET50 - DISSERTATION MODEL CONFIGURATION")
print("=" * 70)

print("\n[1] PRETRAINED WEIGHTS")

print("Pretrained:", PRETRAINED)

if PRETRAINED:
    print("Pretrained source: Check model initialization.")
    print("Expected if using torchvision ImageNet weights:")
    print("    weights=models.ResNet50_Weights.DEFAULT")

print("\n[2] MODEL ARCHITECTURE")

print("Model:", MODEL_NAME)
print("Input image size:", IMAGE_SIZE)
print("Number of classes:", NUM_CLASSES)
print("Class names:", CLASS_NAMES)

print("\n[3] MODEL PARAMETERS")

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

non_trainable_params = total_params - trainable_params

print(f"Total parameters:       {total_params:,}")
print(f"Trainable parameters:   {trainable_params:,}")
print(f"Non-trainable parameters:{non_trainable_params:,}")


print("\n[4] OPTIMIZER")

if "optimizer" in globals():

    print("Optimizer:", optimizer.__class__.__name__)

    for i, param_group in enumerate(optimizer.param_groups):

        print(f"\nParameter group {i + 1}")

        print(
            "Learning rate:",
            param_group.get("lr")
        )

        print(
            "Weight decay:",
            param_group.get("weight_decay")
        )

        print(
            "Momentum:",
            param_group.get("momentum", "N/A")
        )

else:

    print("Optimizer object not found.")
    print("Run this section after creating the optimizer.")

print("\n[5] LOSS FUNCTION")

if "criterion" in globals():

    print(
        "Loss function:",
        criterion.__class__.__name__
    )

    print(criterion)

elif "loss_fn" in globals():

    print(
        "Loss function:",
        loss_fn.__class__.__name__
    )

    print(loss_fn)

elif "loss_function" in globals():

    print(
        "Loss function:",
        loss_function.__class__.__name__
    )

    print(loss_function)

else:

    print("Loss function object not found.")

print("\n[6] LEARNING RATE SCHEDULER")

if "scheduler" in globals():

    print(
        "Scheduler:",
        scheduler.__class__.__name__
    )

    print("Scheduler settings:")

    if hasattr(scheduler, "factor"):
        print("Factor:", scheduler.factor)

    if hasattr(scheduler, "patience"):
        print("Patience:", scheduler.patience)

    if hasattr(scheduler, "min_lrs"):
        print("Minimum LR:", scheduler.min_lrs)

    if hasattr(scheduler, "gamma"):
        print("Gamma:", scheduler.gamma)

    if hasattr(scheduler, "step_size"):
        print("Step size:", scheduler.step_size)

else:

    print("Scheduler object not found.")

print("\n[7] TRAINING CONFIGURATION")

print("Batch size:", BATCH_SIZE)
print("Freeze backbone:", FREEZE_BACKBONE)
print("Freeze epochs:", FREEZE_EPOCHS)
print("Fine-tuning epochs:", FINE_TUNE_EPOCHS)
print("Total epochs:", NUM_EPOCHS)

print("Backbone learning rate:", BACKBONE_LEARNING_RATE)
print("Classifier learning rate:", CLASSIFIER_LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)

print("Label smoothing:", LABEL_SMOOTHING)

print("Early stopping:", USE_EARLY_STOPPING)
print("Early stopping patience:", EARLY_STOPPING_PATIENCE)

print("Model selection metric:", MODEL_SELECTION_METRIC)

print("Scheduler enabled:", USE_SCHEDULER)
print("Scheduler type:", SCHEDULER_TYPE)

print("Scheduler factor:", SCHEDULER_FACTOR)
print("Scheduler patience:", SCHEDULER_PATIENCE)
print("Minimum learning rate:", MIN_LR)

print("\n[8] DATASET CONFIGURATION")

print("Image size:", IMAGE_SIZE)
print("Number of classes:", NUM_CLASSES)
print("Classes:", CLASS_NAMES)

print("Training directory:", TRAIN_DIR)
print("Validation directory:", VAL_DIR)
print("Test directory:", TEST_DIR)

print("\n[9] COMPUTING DEVICE")

print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print("\n" + "=" * 70)
print("END OF MODEL INFORMATION")
print("=" * 70)