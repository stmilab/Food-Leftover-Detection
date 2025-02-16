import torch.nn as nn
import torch
import math
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


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
        )
    )


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "bn_" + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, "bn_" + str(i))(x) for i, x in enumerate(x_parallel)]


class BatchNorm1dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm1dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "bn_" + str(i), nn.BatchNorm1d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, "bn_" + str(i))(x) for i, x in enumerate(x_parallel)]


class LayerNormParallel(nn.Module):
    def __init__(self, normalized_shape, num_parallel):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "ln_" + str(i), nn.LayerNorm(normalized_shape))

    def forward(self, x_parallel):
        return [getattr(self, "ln_" + str(i))(x) for i, x in enumerate(x_parallel)]


class Classifier(nn.Module):
    def __init__(self, input_size=64 * 6, hidden=512, output_size=4, dropout=0.1):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(hidden, hidden // 2)
        self.layer3 = nn.Linear(hidden // 2, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, b_zs, a_zs):
        # Concatenate inputs along the feature dimension
        b_zs = torch.cat([x for x in b_zs], dim=1)
        a_zs = torch.cat([x for x in a_zs], dim=1)
        zs = torch.cat([b_zs, a_zs], dim=1)
        zs = zs.flatten(1)
        zs = self.layer1(zs)
        zs = self.dropout(zs)
        zs = self.act(zs)
        zs = self.layer2(zs)
        zs = self.dropout(zs)
        zs = self.act(zs)
        zs = self.layer3(zs)
        zs = self.softmax(zs)
        return zs


class vanilla_vit(nn.Module):
    def __init__(
        self, img_model, input_size=64, hidden=512, output_size=4, dropout=0.1
    ):
        super().__init__()
        self.model = img_model
        self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(hidden, hidden // 2)
        self.layer3 = nn.Linear(hidden // 2, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        zs = self.model(x)
        zs = zs.flatten(1)
        zs = self.layer1(zs)
        zs = self.dropout(zs)
        zs = self.act(zs)
        zs = self.layer2(zs)
        zs = self.dropout(zs)
        zs = self.act(zs)
        zs = self.layer3(zs)
        zs = self.softmax(zs)
        return zs


class CEN(nn.Module):
    def __init__(
        self,
        before_models,
        after_models,
        num_parallel=3,
        threshold=0,  # by default CEN is disabled
        crop_size=224,
    ):
        super(CEN, self).__init__()
        # NOTE: encoding data using ViT
        self.before_vit = before_models  # 3 models per before
        self.after_vit = after_models  # 3 models per after
        self.conv1 = conv1x1(crop_size, 64)
        self.num_ch = num_parallel
        self.bn = LayerNormParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        # NOTE: Exchanging channels
        self.exchange = Exchange()
        self.threshold = threshold
        self.ln_list = []
        for module in self.ln.modules():
            if isinstance(module, nn.LayerNorm):
                self.ln_list.append(module)
        # NOTE: Merging the values into a singular
        self.classifier = Classifier()

    def forward(self, b_x, a_x):
        b_zs, a_zs = [], []
        for i in range(self.num_ch):
            # NOTE: a_x.shape==b_x.shape==[batch, channel, RGB, H, W],
            b_zs.append(self.after_vit(b_x[:, i]))
            a_zs.append(self.after_vit(a_x[:, i]))
        assert len(a_zs) == self.num_ch, "Input doesn't contain 3 images!!"
        b_zs, a_zs = self.ln(b_zs), self.ln(a_zs)
        b_zs = self.exchange(b_zs, self.ln_list, self.threshold)
        a_zs = self.exchange(a_zs, self.ln_list, self.threshold)
        b_zs, a_zs = self.relu(b_zs), self.relu(a_zs)
        return self.classifier(b_zs, a_zs)


class CEN_CNN(nn.Module):
    def __init__(
        self,
        num_parallel=3,
        threshold=0,  # by default CEN is disabled
        crop_size=224,
    ):
        super(CEN_CNN, self).__init__()
        self.conv1 = conv1x1(3, 32)
        self.bn1 = BatchNorm2dParallel(32, num_parallel)
        self.conv2 = conv1x1(32, 64)
        self.bn2 = BatchNorm2dParallel(64, num_parallel)
        self.conv3 = conv1x1(64, 128)
        self.bn3 = BatchNorm2dParallel(128, num_parallel)
        self.num_ch = num_parallel
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.pool = ModuleParallel(nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = ModuleParallel(nn.Dropout(p=0.3))
        # NOTE: Exchanging channels
        self.exchange = Exchange()
        self.threshold = threshold
        self.bn_list1 = []
        for module in self.bn1.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                self.bn_list1.append(module)
        self.bn_list2 = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                self.bn_list2.append(module)
        # NOTE: Merging the values into a singular
        self.classifier = Classifier(input_size=768 * 28 * 28, hidden=1024)

    def forward(self, b_x, a_x):
        b_x, a_x = b_x.permute(1, 0, 2, 3, 4), a_x.permute(1, 0, 2, 3, 4)
        # NOTE: a_x.shape==b_x.shape==[batch, channel, RGB, H, W],
        b_zs, a_zs = self.conv1(b_x), self.conv1(a_x)
        b_zs, a_zs = self.bn1(b_zs), self.bn1(a_zs)
        b_zs = self.exchange(b_zs, self.bn_list1, self.threshold)
        a_zs = self.exchange(a_zs, self.bn_list1, self.threshold)
        b_zs, a_zs = self.pool(self.relu(b_zs)), self.pool(self.relu(a_zs))

        b_zs, a_zs = self.conv2(b_zs), self.conv2(a_zs)
        b_zs, a_zs = self.bn2(b_zs), self.bn2(a_zs)
        b_zs = self.exchange(b_zs, self.bn_list2, self.threshold)
        a_zs = self.exchange(a_zs, self.bn_list2, self.threshold)
        b_zs, a_zs = self.pool(self.relu(b_zs)), self.pool(self.relu(a_zs))

        b_zs, a_zs = self.conv3(b_zs), self.conv3(a_zs)
        b_zs, a_zs = self.bn3(b_zs), self.bn3(a_zs)
        b_zs, a_zs = self.pool(self.relu(b_zs)), self.pool(self.relu(a_zs))
        b_zs, a_zs = self.dropout(b_zs), self.dropout(a_zs)
        return self.classifier(b_zs, a_zs)
