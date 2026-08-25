import torch
import torch.nn as nn
from torchvision import models
import config
class ResNet50Classifier(nn.Module):

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    ):
        super().__init__()

        if pretrained:
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
        else:
            self.backbone = models.resnet50(
                weights=None
            )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(
                in_features,
                512
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(
                512,
                num_classes
            )
        )

    def freeze_backbone(self):

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def set_head_training_mode(self):

        self.backbone.eval()
        self.backbone.fc.train()

    def unfreeze_backbone(self):

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

def forward(self, x):

        return self.backbone(x)
def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
def set_head_training_mode(self):
    self.train()
    self.backbone.eval()
    self.backbone.fc.train()
def create_model():
    model = ResNet50Classifier()
    model = model.to(
        config.DEVICE
    )
    return model
def count_parameters(model):
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )
    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return (
        total_parameters,
        trainable_parameters
    )
def print_model_info(model):
    total, trainable = count_parameters(model)
    print("\n")
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print(
        "Model:",
        config.MODEL_NAME
    )
    print(
        "Device:",
        config.DEVICE
    )
    print(
        "Total Parameters:",
        f"{total:,}"
    )
    print(
        "Trainable Parameters:",
        f"{trainable:,}"
    )
    print(
        "Frozen Parameters:",
        f"{total - trainable:,}"
    )
    print("=" * 70)
if __name__ == "__main__":
    model = create_model()
    print_model_info(model)
    dummy_input = torch.randn(
        4,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ).to(
        config.DEVICE
    )
    output = model(
        dummy_input
    )
    print("\nForward Pass Test")
    print("-" * 30)
    print(
        "Input shape:",
        dummy_input.shape
    )
    print(
        "Output shape:",
        output.shape
    )