import torch
import numpy as np


class DownsamplingBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 15,
    ):
        super().__init__()

        assert kernel_size % 2 != 0  # kernel must be odd length
        padding = kernel_size // 2  # calculate same padding

        self.conv1 = torch.nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm1d(ch_out)
        self.prelu = torch.nn.PReLU(ch_out)
        self.conv2 = torch.nn.Conv1d(
            ch_out,
            ch_out,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.prelu(x)
        x_ds = self.conv2(x)
        return x_ds, x


class UpsamplingBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        skip: str = "add",
    ):
        super().__init__()

        assert kernel_size % 2 != 0  # kernel must be odd length
        padding = kernel_size // 2  # calculate same padding

        self.skip = skip
        self.conv = torch.nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm1d(ch_out)
        self.prelu = torch.nn.PReLU(ch_out)
        self.us = torch.nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.us(x)  # upsample by x2

        # handle skip connections
        if self.skip == "add":
            x = x + skip
        elif self.skip == "concat":
            x = torch.cat((x, skip), dim=1)
        elif self.skip == "none":
            pass
        else:
            raise NotImplementedError()

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class MixWaveUNet(torch.nn.Module):
    """

    Martínez Ramírez M. A., Stoller, D. and Moffat, D., “A deep learning approach to intelligent drum mixing with the Wave-U-Net”
    Journal of the Audio Engineering Society, vol. 69, no. 3, pp. 142-151, March 2021
    """

    def __init__(
        self,
        ninputs: int,
        noutputs: int,
        ds_kernel: int = 13,
        us_kernel: int = 13,
        out_kernel: int = 5,
        layers: int = 12,
        ch_growth: int = 24,
        skip: str = "concat",
    ):
        super().__init__()
        self.encoder = torch.nn.ModuleList()
        for n in np.arange(layers):

            if n == 0:
                ch_in = ninputs
                ch_out = ch_growth
            else:
                ch_in = ch_out
                ch_out = ch_in + ch_growth

            self.encoder.append(DownsamplingBlock(ch_in, ch_out, kernel_size=ds_kernel))

        self.embedding = torch.nn.Conv1d(ch_out, ch_out, kernel_size=1)

        self.decoder = torch.nn.ModuleList()
        for n in np.arange(layers, stop=0, step=-1):

            ch_in = ch_out
            ch_out = ch_in - ch_growth

            if ch_out < ch_growth:
                ch_out = ch_growth

            if skip == "concat":
                ch_in *= 2

            self.decoder.append(
                UpsamplingBlock(
                    ch_in,
                    ch_out,
                    kernel_size=us_kernel,
                    skip=skip,
                )
            )

        self.output_conv = torch.nn.Conv1d(
            ch_out + ninputs,
            noutputs,
            kernel_size=out_kernel,
            padding=out_kernel // 2,
        )

    def forward(self, x: torch.Tensor, track_mask: torch.Tensor = None):
        x_in = x
        skips = []

        for enc in self.encoder:
            x, skip = enc(x)
            skips.append(skip)

        x = self.embedding(x)

        for dec in self.decoder:
            skip = skips.pop()
            x = dec(x, skip)

        x = torch.cat((x_in, x), dim=1)

        y = self.output_conv(x)

        return y, torch.zeros(1)  # return dummy parameters
