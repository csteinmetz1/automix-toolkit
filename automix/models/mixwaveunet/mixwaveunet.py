import torch
from .waveunet.waveunet import Waveunet


class MixWaveUNet(torch.nn.Module):
    """Mix-WaveUNet

    Martínez Ramírez M. A., Stoller, D. and Moffat, D., “A deep learning approach to intelligent drum mixing with the Wave-U-Net”
    Journal of the Audio Engineering Society, vol. 69, no. 3, pp. 142-151, March 2021
    """

    def __init__(
        self,
        num_tracks: int,
        target_output_size: int,
        num_channels: int = 32,
        levels: int = 6,
    ) -> None:
        super().__init__()
        self.num_tracks = num_tracks
        num_features = [num_channels * i for i in range(1, levels + 1)]
        self.waveunet = Waveunet(
            num_tracks,
            num_features,
            2,
            ["ALL"],
            5,
            target_output_size,
            "gn",
            "fixed",
        )

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): Input tracks with shape (bs, num_tracks, seq_len)

        Returns:
            y (torch.Tensor): Final stereo mixture with shape (bs, 2, seq_len)
        """
        pad_size = self.waveunet.input_size - x.shape[-1]
        x_pad = torch.nn.functional.pad(x, (0, pad_size))
        y = self.waveunet(x_pad)
        y = y["ALL"]
        y = y[..., : x.shape[-1]]
        return y
