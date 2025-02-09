import torch
import torch.nn as nn

VGG_type = {
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_version="VGG16", in_channels=3, num_classes=64):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv(VGG_type[vgg_version])
        # after completing all the conv layer the final matrix will be [bs, 512, 7, 7]
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

    def create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
                in_channels = out_channels
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = VGG()
    x = torch.randn(16, 3, 224, 224)
    print(model(x).shape)
    print(model)
