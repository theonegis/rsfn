import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum

from ssim import msssim
from data import NUM_BANDS


class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()
    MaxPooling = enum.auto()


class Normalization(enum.Enum):
    BatchNorm = enum.auto()
    InstanceNorm = enum.auto()
    LayerNorm = enum.auto()
    SpectralNorm = enum.auto()


def compute_gradient(inputs):
    kernel_v = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]
    kernel_h = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).to(inputs.device)
    kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).to(inputs.device)
    gradients = []
    for i in range(inputs.shape[1]):
        data = inputs[:, i]
        data_v = F.conv2d(data.unsqueeze(1), kernel_v, padding=1)
        data_h = F.conv2d(data.unsqueeze(1), kernel_h, padding=1)
        data = torch.sqrt(torch.pow(data_v, 2) + torch.pow(data_h, 2) + 1e-6)
        gradients.append(data)

    result = torch.cat(gradients, dim=1)
    return result


class ReconstructionLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            self.model.conv1,
            self.model.conv2,
            self.model.conv3,
            self.model.conv4
        )

    @staticmethod
    def safe_arccos(inputs, epsilon=1e-6):
        return torch.acos(torch.clamp(inputs, -1 + epsilon, 1 - epsilon))

    def forward(self, prediction, target):
        sobel_loss = F.mse_loss(compute_gradient(prediction), compute_gradient(target))
        feature_loss = F.mse_loss(self.encoder(prediction), self.encoder(target))
        spectral_loss = torch.mean(self.safe_arccos(F.cosine_similarity(prediction, target, 1)))
        vision_loss = 1.0 - msssim(prediction, target, normalize=True)
        loss = sobel_loss + feature_loss + spectral_loss + vision_loss
        return loss


class Resample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, 3, stride=stride, padding=1)


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


def norma_layer(norm, channels):
    mapping = {
        Normalization.BatchNorm: nn.BatchNorm2d(channels),
        Normalization.InstanceNorm: nn.InstanceNorm2d(channels),
        Normalization.LayerNorm: nn.LayerNorm(channels)
    }
    return mapping[norm]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None, norm=None):
        super().__init__()
        channels = min(in_channels, out_channels)
        residual = [
            Conv3X3WithPadding(in_channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1)
        ]
        transform = [nn.Conv2d(in_channels, out_channels, 1)]

        if sampling == Sampling.UpSampling:
            residual.insert(0, Resample(2))
            transform.insert(0, Resample(2))
        elif sampling == Sampling.DownSampling:
            residual[0] = Conv3X3WithPadding(in_channels, channels, 2)

        if norm is not None:
            residual.insert(-2, norma_layer(norm, channels))
            residual.append(norma_layer(norm, out_channels))
            transform.append(norma_layer(norm, out_channels))

        self.residual = nn.Sequential(*residual)
        self.transform = transform[0] if len(transform) == 1 else nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        lateral = self.transform(inputs)
        if lateral.shape != trunk.shape:
            lateral = F.interpolate(lateral, size=trunk.shape[-2:], mode='bilinear', align_corners=False)
        return trunk + lateral


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        super().__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ResidualBlock(in_channels, channels[0])
        self.conv2 = ResidualBlock(channels[0], channels[1], sampling=Sampling.DownSampling)
        self.conv3 = ResidualBlock(channels[1], channels[2], sampling=Sampling.DownSampling)
        self.conv4 = ResidualBlock(channels[2], channels[3], sampling=Sampling.DownSampling)
        self.conv5 = ResidualBlock(channels[3], channels[2], sampling=Sampling.UpSampling)
        self.conv6 = ResidualBlock(channels[2] * 2, channels[1], sampling=Sampling.UpSampling)
        self.conv7 = ResidualBlock(channels[1] * 2, channels[0], sampling=Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        out = self.conv8(torch.cat((l1, l7), 1))
        return out


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = (16, 32, 64)

        self.fencoder = nn.Sequential(
            ResidualBlock(NUM_BANDS, self.channels[0]),
            ResidualBlock(self.channels[0], self.channels[1]),
            ResidualBlock(self.channels[1], self.channels[2], sampling=Sampling.UpSampling)
        )

        self.cencoder = nn.Sequential(
            ResidualBlock(NUM_BANDS, self.channels[0]),
            ResidualBlock(self.channels[0], self.channels[1]),
            ResidualBlock(self.channels[1], self.channels[2])
        )

        self.attention = nn.Sequential(
            ResidualBlock(NUM_BANDS * 2, self.channels[0],
                          norm=Normalization.InstanceNorm, sampling=Sampling.DownSampling),
            ResidualBlock(self.channels[0], self.channels[1],
                          norm=Normalization.InstanceNorm, sampling=Sampling.DownSampling),
            ResidualBlock(self.channels[1], self.channels[2],
                          norm=Normalization.InstanceNorm, sampling=Sampling.DownSampling),
            ResidualBlock(self.channels[2], self.channels[2],
                          norm=Normalization.InstanceNorm, sampling=Sampling.UpSampling),
            ResidualBlock(self.channels[2], self.channels[1], sampling=Sampling.UpSampling,
                          norm=Normalization.InstanceNorm),
            ResidualBlock(self.channels[1], self.channels[0], sampling=Sampling.UpSampling,
                          norm=Normalization.InstanceNorm),
            ResidualBlock(self.channels[0], 1, sampling=Sampling.UpSampling,
                          norm=Normalization.InstanceNorm),
            nn.Sigmoid()
        )

        self.tweak = ResidualBlock(self.channels[2], self.channels[2], sampling=Sampling.DownSampling)

        self.decoder = nn.Sequential(
            ResidualBlock(self.channels[2] * 2, self.channels[2]),
            ResidualBlock(self.channels[2], self.channels[1]),
            ResidualBlock(self.channels[1], self.channels[0]),
            nn.Conv2d(self.channels[0], NUM_BANDS, 1)
        )

    def forward(self, inputs):
        prev_fine, next_fine = self.fencoder(inputs[0]), self.fencoder(inputs[1])
        coarse = self.cencoder(inputs[-1])

        prev_attn = self.attention(torch.cat((inputs[0], inputs[-1]), 1))
        next_attn = self.attention(torch.cat((inputs[1], inputs[-1]), 1))
        if prev_attn.shape[-2:] != prev_fine.shape[-2:]:
            prev_attn = F.interpolate(prev_attn, prev_fine.shape[-2:], mode='bilinear', align_corners=False)
            next_attn = F.interpolate(next_attn, next_fine.shape[-2:], mode='bilinear', align_corners=False)
        reference = prev_fine * prev_attn+ next_fine * next_attn
        reference = self.tweak(reference)

        prediction = self.decoder(torch.cat((reference, coarse), 1))
        return prediction


class DResidualBlockWithSN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(
                Conv3X3NoPadding(in_channels, in_channels)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(
                nn.Conv2d(in_channels, out_channels, 1)),
        )
        self.transform = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1)),
            nn.MaxPool2d(2)
        )

    def forward(self, inputs):
        return self.transform(inputs) + self.residual(inputs)


# 判别器网络
class Discriminator(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 2, 64, 64, 128, 128, 256, 256]
        modules = []
        for i in range(1, (len(channels))):
            modules.append(DResidualBlockWithSN(channels[i - 1], channels[i]))
        modules.append(SpectralNorm2d(nn.Conv2d(channels[-1], 1, 1)))
        super().__init__(*modules)

    def forward(self, inputs):
        prediction = super().forward(inputs)
        return prediction.view(-1, 1).squeeze(1)