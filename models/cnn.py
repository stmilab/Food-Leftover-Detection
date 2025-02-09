import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2, bn3 = bn[0].weight.abs(), bn[1].weight.abs(), bn[2].weight.abs()
        x1, x2, x3 = x[0].clone(), x[1].clone(), x[2].clone()
        new_bn = torch.max(torch.stack([x[0], x[1], x[2]]), dim=0).values
        mask1 = bn1 < bn_threshold
        mask2 = bn2 < bn_threshold
        mask3 = bn3 < bn_threshold
        x1[:, mask1] = new_bn[:, mask1]
        x2[:, mask2] = new_bn[:, mask2]
        x3[:, mask3] = new_bn[:, mask3]
        return [x1, x2, x3]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "bn_" + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, "bn_" + str(i))(x) for i, x in enumerate(x_parallel)]


class CustomCNN(nn.Module):
    def __init__(self, num_classes=64, num_parallel=3, cen_mode=False):
        super(CustomCNN, self).__init__()
        self.cen_mode = cen_mode
        self.num_ch = num_parallel
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = (
            BatchNorm2dParallel(32, num_parallel) if cen_mode else nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = (
            BatchNorm2dParallel(64, num_parallel) if cen_mode else nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = (
            BatchNorm2dParallel(128, num_parallel) if cen_mode else nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = (
            BatchNorm2dParallel(256, num_parallel) if cen_mode else nn.BatchNorm2d(256)
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.exchange = Exchange()
        self.threshold = 1e-3
        self.bn_list = []
        for module in self.bn.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                self.bn_list.append(module)
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Flattened size depends on input size
        self.fc2 = nn.Linear(512, num_classes)  # Output 64 logits

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        pdb.set_trace()
        if self.cen_mode:
            x = self.cen_forward(x)
        else:
            x = self.pool(
                F.relu(self.bn1(self.conv1(x)))
            )  # Conv1 -> BatchNorm -> ReLU -> Pool
            x = self.pool(
                F.relu(self.bn2(self.conv2(x)))
            )  # Conv2 -> BatchNorm -> ReLU -> Pool
            x = self.pool(
                F.relu(self.bn3(self.conv3(x)))
            )  # Conv3 -> BatchNorm -> ReLU -> Pool
            x = self.pool(
                F.relu(self.bn4(self.conv4(x)))
            )  # Conv4 -> BatchNorm -> ReLU -> Pool

        x = torch.flatten(x, start_dim=1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Fully connected layer 2 (Output layer)

        return x
