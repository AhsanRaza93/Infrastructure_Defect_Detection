from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import config

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

def get_image_files(directory):
    directory = Path(directory)
    files = [
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort()
    return files

def find_mask(image_path, mask_directory):
    image_path = Path(image_path)
    mask_directory = Path(mask_directory)
    candidates = [
        mask_directory / (
            image_path.stem + extension
        )
        for extension in IMAGE_EXTENSIONS
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "\nMask not found for image:\n"
        f"{image_path}\n\n"
        "Expected mask with the same filename stem "
        f"inside:\n{mask_directory}"
    )

class CrackSegmentationDataset(Dataset):
    """
    Dataset for binary crack segmentation.
    Returns:
        image:
            Tensor [3, H, W]
        mask:
            Tensor [1, H, W]
        image_path:
            Original image path
    """
    def __init__(
        self,
        image_directory,
        mask_directory,
        image_size=None,
    ):
        self.image_directory = Path(
            image_directory
        )
        self.mask_directory = Path(
            mask_directory
        )
        self.image_size = (
            image_size
            if image_size is not None
            else config.IMAGE_SIZE
        )
        self.image_files = get_image_files(
            self.image_directory
        )
        if len(self.image_files) == 0:
            raise RuntimeError(
                "\nNo images were found in:\n"
                f"{self.image_directory}"
            )
        self.mask_files = []
        for image_path in self.image_files:
            mask_path = find_mask(
                image_path,
                self.mask_directory,
            )
            self.mask_files.append(
                mask_path
            )
        if len(self.image_files) != len(
            self.mask_files
        ):
            raise RuntimeError(
                "Number of images and masks do not match."
            )
    
    def __len__(self):
        return len(self.image_files)
    
    def load_image(self, image_path):
        """
        Load and preprocess RGB image.
        """
        image = Image.open(
            image_path
        ).convert("RGB")
        image = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.BILINEAR,
        )
        image = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0
        image = np.transpose(
            image,
            (2, 0, 1),
        )
        image = torch.from_numpy(
            image
        )
        mean = torch.tensor(
            config.IMAGENET_MEAN,
            dtype=image.dtype,
        ).view(
            3,
            1,
            1,
        )
        std = torch.tensor(
            config.IMAGENET_STD,
            dtype=image.dtype,
        ).view(
            3,
            1,
            1,
        )
        image = (
            image - mean
        ) / std
        return image
    
    def load_mask(self, mask_path):
        """
        Load binary segmentation mask.
        Any non-zero pixel is considered crack.
        """
        mask = Image.open(
            mask_path
        ).convert("L")
        mask = mask.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(
            mask,
            dtype=np.float32,
        )
        mask = (
            mask > 0
        ).astype(
            np.float32
        )
        mask = torch.from_numpy(
            mask
        )
        mask = mask.unsqueeze(0)
        return mask

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]
        image = self.load_image(
            image_path
        )
        mask = self.load_mask(
            mask_path
        )
        return (
            image,
            mask,
            str(image_path),
        )

def get_train_dataset():
    return CrackSegmentationDataset(
        image_directory=config.TRAIN_IMAGE_DIR,
        mask_directory=config.TRAIN_MASK_DIR,
        image_size=config.IMAGE_SIZE,
    )

def get_val_dataset():
    return CrackSegmentationDataset(
        image_directory=config.VAL_IMAGE_DIR,
        mask_directory=config.VAL_MASK_DIR,
        image_size=config.IMAGE_SIZE,
    )

def get_test_dataset():
    return CrackSegmentationDataset(
        image_directory=config.TEST_IMAGE_DIR,
        mask_directory=config.TEST_MASK_DIR,
        image_size=config.IMAGE_SIZE,
    )

if __name__ == "__main__":
    config.validate_dataset_directories()
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()
    test_dataset = get_test_dataset()
    print("\nDataset check")
    print("=" * 60)
    print(
        f"Training images   : {len(train_dataset):,}"
    )
    print(
        f"Validation images : {len(val_dataset):,}"
    )
    print(
        f"Test images       : {len(test_dataset):,}"
    )
    image, mask, path = train_dataset[0]
    print(
        f"\nImage shape : {tuple(image.shape)}"
    )
    print(
        f"Mask shape  : {tuple(mask.shape)}"
    )
    print(
        f"Image path  : {path}"
    )