from pathlib import Path
import shutil
from ultralytics import YOLO
import config
from dataset import validate_dataset
from model import load_base_model
def train_model():
    """
    Train YOLOv8 using the train and validation splits.
    The test split is NOT used during training.
    """
    print("\n" + "=" * 80)
    print("YOLOv8 CIVIL INFRASTRUCTURE DEFECT DETECTION")
    print("TRAINING")
    print("=" * 80)
    print(f"Dataset YAML : {config.DATA_YAML}")
    print(f"Image size   : {config.IMAGE_SIZE}")
    print(f"Epochs       : {config.EPOCHS}")
    print(f"Batch size   : {config.BATCH_SIZE}")
    print(f"Device       : {config.DEVICE}")
    model = load_base_model()
    results = model.train(
        data=str(config.DATA_YAML),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        workers=config.WORKERS,
        device=config.DEVICE,
        seed=config.SEED,
        project=str(config.RESULTS_DIR),
        name=config.PROJECT_NAME,
        exist_ok=True,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    return model, results
def copy_best_checkpoint():
    """
    Copy Ultralytics best.pt to the project's checkpoint folder.
    """
    training_directory = (
        config.RESULTS_DIR /
        config.PROJECT_NAME
    )
    source = (
        training_directory /
        "weights" /
        "best.pt"
    )
    destination = config.BEST_MODEL_PATH
    if not source.exists():
        raise FileNotFoundError(
            "\nUltralytics best.pt was not found:\n"
            f"{source}"
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
        "\nBest model copied to:"
    )
    print(destination)
    return destination
def main():
    config.create_directories()
    print("\nStep 1: Validating dataset...")
    validate_dataset()
    print("\nStep 2: Training YOLOv8...")
    train_model()
    print("\nStep 3: Saving best checkpoint...")
    copy_best_checkpoint()
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
if __name__ == "__main__":
    main()