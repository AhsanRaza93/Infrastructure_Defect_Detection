import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import config
from model import create_model
prediction_transform = transforms.Compose([
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
def load_model():
    model = create_model()
    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE,
        weights_only=False
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    return model
def predict_image(
    image_path
):
    model = load_model()
    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )
    image_tensor = prediction_transform(
        image
    )
    image_tensor = image_tensor.unsqueeze(
        0
    )
    image_tensor = image_tensor.to(
        config.DEVICE
    )
    with torch.no_grad():
        output = model(
            image_tensor
        )
        probabilities = F.softmax(
            output,
            dim=1
        )
        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )
    predicted_index = prediction.item()
    predicted_class = (
        config.CLASS_NAMES[
            predicted_index
        ]
    )
    confidence_value = (
        confidence.item()
        * 100
    )
    crack_probability = (
        probabilities[0, 0].item()
        * 100
    )
    no_crack_probability = (
        probabilities[0, 1].item()
        * 100
    )
    return {
        "class":
            predicted_class,
        "confidence":
            confidence_value,
        "crack_probability":
            crack_probability,
        "no_crack_probability":
            no_crack_probability
    }
from pathlib import Path

image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

test_images = [
    p for p in Path(config.TEST_DIR).rglob("*")
    if p.suffix.lower() in image_extensions
]

print("Number of test images:", len(test_images))
print("First image:", test_images[29])
if __name__ == "__main__":
    IMAGE_PATH = "Paste image path here"
    result = predict_image(
        IMAGE_PATH
    )
    print("\n")
    print("=" * 60)
    print("CIVIL INFRASTRUCTURE CRACK CLASSIFICATION")
    print("=" * 60)
    print(
        f"Prediction          : "
        f"{result['class']}"
    )
    print(
        f"Confidence          : "
        f"{result['confidence']:.2f}%"
    )
    print(
        f"Crack Probability   : "
        f"{result['crack_probability']:.2f}%"
    )
    print(
        f"No-crack Probability: "
        f"{result['no_crack_probability']:.2f}%"
    )
    print("=" * 60)