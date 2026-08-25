import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet34,
    ResNet34_Weights,
)
import config

class ConvBlock(nn.Module):
    """
    Two convolution layers followed by BatchNorm and ReLU.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    """
    U-Net decoder block.
    Upsamples decoder feature map, concatenates skip
    connection, and applies convolutional refinement.
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )
    def forward(
        self,
        x,
        skip,
    ):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat(
            [x, skip],
            dim=1,
        )
        x = self.conv(x)
        return x

class UNetResNet34(nn.Module):
    """
    U-Net with ResNet34 encoder.
    Encoder feature channels:
        64
        64
        128
        256
        512
    Decoder:
        256
        128
        64
        32
        16
    """
    def __init__(
        self,
        pretrained=True,
        num_classes=1,
    ):
        super().__init__()
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None
        backbone = resnet34(
            weights=weights
        )
        self.input_block = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4
        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )
        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )
        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )
        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=32,
        )
        self.final_decoder = nn.Sequential(
            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                16,
                num_classes,
                kernel_size=1,
            ),
        )
    def forward(self, x):
        input_size = x.shape[-2:]
        x0 = self.input_block(x)
        # 64 channels, H/2
        x1 = self.encoder1(
            self.pool(x0)
        )
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        d4 = self.decoder4(
            x4,
            x3,
        )
        d3 = self.decoder3(
            d4,
            x2,
        )
        d2 = self.decoder2(
            d3,
            x1,
        )
        d1 = self.decoder1(
            d2,
            x0,
        )
        d1 = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        output = self.final_decoder(
            d1
        )
        return output

def create_model(
    pretrained=None,
):
    """
    Create U-Net + ResNet34 model.
    """
    if pretrained is None:
        pretrained = config.PRETRAINED_ENCODER
    model = UNetResNet34(
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
    )
    return model

if __name__ == "__main__":
    model = create_model(
        pretrained=False
    )
    dummy_input = torch.randn(
        2,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    )
    with torch.no_grad():
        output = model(
            dummy_input
        )
    print(
        "Input shape :",
        tuple(dummy_input.shape),
    )
    print(
        "Output shape:",
        tuple(output.shape),
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )