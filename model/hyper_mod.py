import random
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from model.stylegan import PixelNorm, equal_lr, EqualLinear, EqualConv2d, ConstantInput, Blur, NoiseInjection,\
    AdaptiveInstanceNorm, ConvBlock


class SequentialWithParams(nn.Sequential):
    def forward(self, x):
        x, task = x
        for m in self:
            if isinstance(m, (AdaptiveFilterModulation, FusedUpsampleAdaFM)):
                x = m(x, task)
            else:
                x = m(x)

        return x


def equalize(weight):
    """Batched equalization. Since all kernels have the same shape we can apply the scaling to all batches."""
    fan_in = weight[0].data.size(1) * weight[0].data[0][0].numel()
    weight *= sqrt(2 / fan_in)


class AdaptiveFilterModulation(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, task_dim=0):
        super().__init__()
        self.stride = stride
        self.padding = padding

        # these two will be normalized and initialized when loading from a pretrained model
        self.register_buffer('W', torch.empty((out_channel, in_channel, kernel_size, kernel_size)))
        self.register_buffer('b', torch.empty(out_channel))

        self.task = EqualLinear(task_dim, out_channel * in_channel * 2)
        self.task.linear.bias.data[:] = 0

        self.task_bias_beta = EqualLinear(task_dim, out_channel)
        self.task_bias_beta.linear.bias.data[:] = 0

    def forward(self, x, task):
        gamma, beta = self.task(task).view(-1, self.W.shape[0], self.W.shape[1], 2).unsqueeze(4).chunk(2, 3)
        bias_beta = self.task_bias_beta(task)

        W_i = self.W * gamma + beta
        equalize(W_i)
        b_i = self.b + bias_beta
        out = conv1to1(x, W_i, bias=b_i, stride=self.stride, padding=self.padding)
        return out


class FusedUpsampleAdaFM(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=2, padding=0, task_dim=0):
        super().__init__()
        self.stride = stride
        self.padding = padding

        self.register_buffer('W', torch.empty((in_channel, out_channel, kernel_size, kernel_size)))
        self.register_buffer('b', torch.empty(out_channel))

        self.task = EqualLinear(task_dim, out_channel * in_channel * 2)
        self.task.linear.bias.data[:] = 0

        self.task_bias_beta = EqualLinear(task_dim, out_channel)
        self.task_bias_beta.linear.bias.data[:] = 0

    def forward(self, x, task):
        weight = F.pad(self.W, [1, 1, 1, 1])
        weight = (
                         weight[:, :, 1:, 1:]
                         + weight[:, :, :-1, 1:]
                         + weight[:, :, 1:, :-1]
                         + weight[:, :, :-1, :-1]
                 ) / 4

        gamma, beta = self.task(task).view(-1, weight.shape[0], weight.shape[1], 2).unsqueeze(4).chunk(2, 3)
        bias_beta = self.task_bias_beta(task)

        W_i = weight * gamma + beta
        equalize(W_i)
        b_i = self.b + bias_beta
        out = conv1to1(x, W_i, bias=b_i, stride=self.stride, padding=self.padding, transposed=True, mode='upsample')
        return out


def conv1to1(x, weights, bias=None, stride=1, padding=0, transposed=False, mode='normal'):
    """Apply a different kernel to a different sample. result[i] = conv(input[i], weights[i]).
    From https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/3"""
    F_conv = F.conv_transpose2d if transposed else F.conv2d
    N, C, H, W = x.shape
    KN, KO, KI, KH, KW = weights.shape
    assert N == KN

    # group weights
    weights = weights.view(-1, KI, KH, KW)
    bias = bias.view(-1)

    # move batch dim into channels
    x = x.view(1, -1, H, W)

    # Apply grouped conv
    outputs_grouped = F_conv(x, weights, bias=bias, stride=stride, padding=padding, groups=N)
    if mode == 'upsample' or mode == 'downsample' or (outputs_grouped.shape[2] == 1 and outputs_grouped.shape[3] == 1):
        outputs_grouped = outputs_grouped.view(N, -1, outputs_grouped.shape[2], outputs_grouped.shape[3])
    else:
        outputs_grouped = outputs_grouped.view(N, KO, H, W)
    return outputs_grouped


class StyledConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            style_dim=512,
            initial=False,
            upsample=False,
            fused=False,
            task_dim=512,
    ):
        super().__init__()

        if initial:
            self.conv1 = SequentialWithParams(
                ConstantInput(in_channel)
            )

        else:
            if upsample:
                if fused:
                    self.conv1 = SequentialWithParams(
                        FusedUpsampleAdaFM(in_channel, out_channel, kernel_size, padding=padding, task_dim=task_dim),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = SequentialWithParams(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        AdaptiveFilterModulation(in_channel, out_channel, kernel_size, padding=padding, task_dim=task_dim),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = SequentialWithParams(
                    AdaptiveFilterModulation(in_channel, out_channel, kernel_size, padding=padding, task_dim=task_dim))

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = AdaptiveFilterModulation(out_channel, out_channel, kernel_size, padding=padding, task_dim=task_dim)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise, task):
        out = self.conv1((input, task))
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out, task)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, task_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True, task_dim=task_dim),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True, task_dim=task_dim),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True, task_dim=task_dim),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True, task_dim=task_dim),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True, task_dim=task_dim),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused, task_dim=task_dim),  # 128
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused, task_dim=task_dim),  # 256
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused, task_dim=task_dim),  # 512
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused, task_dim=task_dim),  # 1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1), task=None, return_hierarchical=False):
        out = noise[0]
        hierarchical_out = []

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            out = conv(out, style_step, noise[i], task)
            if return_hierarchical:
                hierarchical_out.append(out)

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        if return_hierarchical:
            return out, hierarchical_out
        else:
            return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8, task_dim=512):
        super().__init__()

        self.generator = Generator(code_dim, task_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(
            self,
            input,
            noise=None,
            step=0,
            alpha=-1,
            mean_style=None,
            style_weight=0,
            mixing_range=(-1, -1),
            task=None,
            return_hierarchical=False,
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range, task=task,
                              return_hierarchical=return_hierarchical)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Task(nn.Module):
    def __init__(self, code_dim=512, n_mlp=4, num_labels=0):
        super().__init__()

        layers = [equal_lr(nn.Embedding(num_labels, code_dim))]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.task = nn.Sequential(*layers)

    def forward(self, x):
        return self.task(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        self.n_layer = len(self.progression)

        self.final = EqualConv2d(512, num_classes, 1)

    def forward(self, input, c, step=0, alpha=-1, return_hierarchical=False):
        hierarchical_out = []
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)
            if return_hierarchical:
                hierarchical_out.append(out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out


        out = self.final(out)
        out = out[torch.arange(out.size(0)), c].squeeze(-1)

        if return_hierarchical:
            hierarchical_out.append(out)
            return hierarchical_out

        return out
