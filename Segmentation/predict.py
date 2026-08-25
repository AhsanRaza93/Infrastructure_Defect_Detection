import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import config
from model import create_model

def normalize_image(image):
    image = image.resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
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
    tensor = torch.from_numpy(
        image
    )
    mean = torch.tensor(
        config.IMAGENET_MEAN,
        dtype=tensor.dtype,
    ).view(
        3,
        1,
        1,
    )
    std = torch.tensor(
        config.IMAGENET_STD,
        dtype=tensor.dtype,
    ).view(
        3,
        1,
        1,
    )
    tensor = (
        tensor - mean
    ) / std
    return tensor

def load_model():
    checkpoint_path = (
        config.BEST_MODEL_PATH
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nBest model checkpoint was not found:\n"
            f"{checkpoint_path}\n\n"
            "Run train.py first."
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.DEVICE,
    )
    model = create_model(
        pretrained=False
    )
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )
    model = model.to(
        config.DEVICE
    )
    model.eval()
    print(
        f"\nCheckpoint loaded:\n"
        f"{checkpoint_path}"
    )
    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )
    return model

def create_overlay(
    image,
    prediction_mask,
    alpha=0.45,
):
    image = image.astype(
        np.float32
    )
    prediction_mask = (
        np.squeeze(
            prediction_mask
        ).astype(bool)
    )
    overlay = image.copy()
    red = np.array(
        [255, 0, 0],
        dtype=np.float32,
    )
    overlay[prediction_mask] = (
        (1.0 - alpha)
        * overlay[prediction_mask]
        + alpha * red
    )
    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

def create_probability_heatmap(
    image,
    probability_map,
):
    figure, axis = plt.subplots(
        figsize=(8, 6)
    )
    axis.imshow(
        image
    )
    heatmap = axis.imshow(
        probability_map,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axis.axis(
        "off"
    )
    figure.colorbar(
        heatmap,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Crack Probability",
    )
    figure.tight_layout()
    return figure

def save_binary_mask(
    prediction_mask,
    output_path,
):
    mask = (
        prediction_mask
        .astype(np.uint8)
        * 255
    )
    Image.fromarray(
        mask
    ).save(
        output_path
    )

def save_probability_map(
    probability_map,
    output_path,
):
    probability = np.clip(
        probability_map,
        0,
        1,
    )
    probability = (
        probability * 255
    ).astype(
        np.uint8
    )
    Image.fromarray(
        probability
    ).save(
        output_path
    )

@torch.no_grad()
def predict_image(
    model,
    image_path,
    threshold,
):
    image_path = Path(
        image_path
    )
    if not image_path.exists():
        raise FileNotFoundError(
            "\nInput image was not found:\n"
            f"{image_path}"
        )
    original = Image.open(
        image_path
    ).convert(
        "RGB"
    )
    original_array = np.asarray(
        original
    )
    original_width, original_height = (
        original.size
    )
    tensor = normalize_image(
        original
    )
    tensor = (
        tensor
        .unsqueeze(0)
        .to(config.DEVICE)
    )
    logits = model(
        tensor
    )
    probabilities = torch.sigmoid(
        logits
    )
    probability_map = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )
    probability_image = Image.fromarray(
        (
            probability_map * 255
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )
    )
    probability_image = (
        probability_image.resize(
            (
                original_width,
                original_height,
            ),
            Image.Resampling.BILINEAR,
        )
    )
    probability_map = (
        np.asarray(
            probability_image,
            dtype=np.float32,
        )
        / 255.0
    )
    prediction_mask = (
        probability_map
        >= threshold
    )
    return (
        original_array,
        probability_map,
        prediction_mask,
    )

def save_prediction_outputs(
    original_image,
    probability_map,
    prediction_mask,
    output_directory,
    image_stem,
):
    output_directory = Path(
        output_directory
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    mask_path = (
        output_directory
        / f"{image_stem}_mask.png"
    )
    save_binary_mask(
        prediction_mask,
        mask_path,
    )
    probability_path = (
        output_directory
        / f"{image_stem}_probability.png"
    )
    save_probability_map(
        probability_map,
        probability_path,
    )
    overlay = create_overlay(
        original_image,
        prediction_mask,
    )
    overlay_path = (
        output_directory
        / f"{image_stem}_overlay.png"
    )
    Image.fromarray(
        overlay
    ).save(
        overlay_path
    )
    heatmap_path = (
        output_directory
        / f"{image_stem}_heatmap.png"
    )
    figure = (
        create_probability_heatmap(
            original_image,
            probability_map,
        )
    )
    figure.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(
        figure
    )
    return {
        "mask": mask_path,
        "probability": probability_path,
        "overlay": overlay_path,
        "heatmap": heatmap_path,
    }

def print_prediction_statistics(
    probability_map,
    prediction_mask,
    threshold,
):
    total_pixels = (
        prediction_mask.size
    )
    crack_pixels = int(
        prediction_mask.sum()
    )
    crack_percentage = (
        crack_pixels
        / total_pixels
        * 100.0
    )
    mean_probability = float(
        probability_map.mean()
    )
    maximum_probability = float(
        probability_map.max()
    )
    print(
        "\n" + "=" * 60
    )
    print(
        "PREDICTION STATISTICS"
    )
    print(
        "=" * 60
    )
    print(
        f"Threshold           : "
        f"{threshold:.3f}"
    )
    print(
        f"Total pixels        : "
        f"{total_pixels:,}"
    )
    print(
        f"Predicted crack     : "
        f"{crack_pixels:,} pixels"
    )
    print(
        f"Crack area          : "
        f"{crack_percentage:.2f}%"
    )
    print(
        f"Mean probability    : "
        f"{mean_probability:.4f}"
    )
    print(
        f"Maximum probability : "
        f"{maximum_probability:.4f}"
    )
    print(
        "=" * 60
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Single-image concrete crack "
            "segmentation using "
            "U-Net + ResNet34."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default=None,
        help="Path to input image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory for prediction outputs."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Prediction threshold. "
            "Default: config value."
        ),
    )
    return parser.parse_args()

def main():
    print(
        "\n" + "=" * 70
    )
    print(
        "CONCRETE CRACK SEGMENTATION"
    )
    print(
        "U-NET + RESNET34 SINGLE-IMAGE PREDICTION"
    )
    print(
        "=" * 70
    )
    args = parse_arguments()

    image_path = Path(
        "Paste your path here"
    )

    if args.threshold is None:
        threshold = (
            config.PREDICTION_THRESHOLD
        )
    else:
        threshold = args.threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )
    if args.output is None:
        output_directory = (
            config.PREDICTIONS_DIR
        )
    else:
        output_directory = Path(
            args.output
        )
    print(
        f"\nInput image:\n"
        f"{image_path}"
    )
    print(
        f"\nDevice:\n"
        f"{config.DEVICE}"
    )
    print(
        f"\nThreshold:\n"
        f"{threshold:.3f}"
    )
    model = load_model()
    print(
        "\nRunning segmentation..."
    )
    (
        original_image,
        probability_map,
        prediction_mask,
    ) = predict_image(
        model=model,
        image_path=image_path,
        threshold=threshold,
    )
    print_prediction_statistics(
        probability_map,
        prediction_mask,
        threshold,
    )
    output_paths = (
        save_prediction_outputs(
            original_image=original_image,
            probability_map=probability_map,
            prediction_mask=prediction_mask,
            output_directory=output_directory,
            image_stem=image_path.stem,
        )
    )
    print(
        "\nPrediction outputs saved:"
    )
    print(
        f"Binary mask        : "
        f"{output_paths['mask']}"
    )
    print(
        f"Probability map    : "
        f"{output_paths['probability']}"
    )
    print(
        f"Segmentation       : "
        f"{output_paths['overlay']}"
    )
    print(
        f"Probability heatmap: "
        f"{output_paths['heatmap']}"
    )
    print(
        "\nPrediction completed successfully."
    )
if __name__ == "__main__":
    main()