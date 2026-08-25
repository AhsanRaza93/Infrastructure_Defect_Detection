import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
sys.path.append("Paste your path here")

import config

def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

base_transform = transforms.Compose([
    transforms.Resize(
        256,
        antialias=True
    ),
    transforms.CenterCrop(
        config.IMAGE_SIZE
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])
def create_datasets():
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=base_transform
    )
    val_dataset = datasets.ImageFolder(
        root=config.VAL_DIR,
        transform=base_transform
    )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=base_transform
    )
    # Verify Class Mapping
    expected_classes = [
        "crack",
        "no-crack"
    ]
    expected_mapping = {
        "crack": 0,
        "no-crack": 1
    }
    for name, dataset in [
        ("Training", train_dataset),
        ("Validation", val_dataset),
        ("Testing", test_dataset)
    ]:
        if dataset.classes != expected_classes:
            raise ValueError(
                f"{name} dataset class order is incorrect.\n"
                f"Expected: {expected_classes}\n"
                f"Found: {dataset.classes}\n\n"
                f"Make sure the folders are exactly:\n"
                f"crack/\n"
                f"no-crack/"
            )
        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"{name} dataset class mapping is incorrect.\n"
                f"Expected: {expected_mapping}\n"
                f"Found: {dataset.class_to_idx}"
            )
    return (
        train_dataset,
        val_dataset,
        test_dataset
    )

def create_dataloaders():
    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    )
def get_class_distribution(dataset):
    distribution = {
        class_name: 0
        for class_name in dataset.classes
    }
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        distribution[class_name] += 1
    return distribution
def print_dataset_info():
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()
    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print("\nClasses:")
    print(train_dataset.classes)
    print("\nClass Mapping:")
    print(train_dataset.class_to_idx)
    print("\nTraining Distribution:")
    print(
        get_class_distribution(train_dataset)
    )
    print("\nValidation Distribution:")
    print(
        get_class_distribution(val_dataset)
    )
    print("\nTesting Distribution:")
    print(
        get_class_distribution(test_dataset)
    )
    print("\nTotal Images")
    print("-" * 30)
    print(
        "Training   :",
        len(train_dataset)
    )
    print(
        "Validation :",
        len(val_dataset)
    )
    print(
        "Testing    :",
        len(test_dataset)
    )
    print("\nBatch Information")
    print("-" * 30)
    print(
        "Batch Size:",
        config.BATCH_SIZE
    )
    print(
        "Train batches:",
        len(train_loader)
    )
    print(
        "Validation batches:",
        len(val_loader)
    )
    print(
        "Test batches:",
        len(test_loader)
    )
    print("\nImage Size:")
    print(
        f"{config.IMAGE_SIZE} x {config.IMAGE_SIZE}"
    )
    print("\nData Augmentation:")
    print("Disabled")
    print("=" * 70)
if __name__ == "__main__":
    set_seed()
    print_dataset_info()
    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_dataloaders()
    images, labels = next(
        iter(train_loader)
    )
    print("\nBatch Verification")
    print("-" * 30)
    print(
        "Image tensor:",
        images.shape
    )
    print(
        "Label tensor:",
        labels.shape
    )
    print(
        "Labels in batch:",
        labels[:10].tolist()
    )