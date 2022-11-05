import torch
import torchaudio

from automix.utils import restore_from_0to1


class Res_2d(torch.nn.Module):
    """Residual 2D Convolutional layer.

    Args:
        input_channels (int):

    Adapted from https://github.com/minzwon/sota-music-tagging-models. Licensed under MIT by Minz Won.
    """

    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = torch.nn.Conv2d(
            input_channels,
            output_channels,
            shape,
            stride=stride,
            padding=shape // 2,
        )
        self.bn_1 = torch.nn.BatchNorm2d(output_channels)
        self.conv_2 = torch.nn.Conv2d(
            output_channels,
            output_channels,
            shape,
            padding=shape // 2,
        )
        self.bn_2 = torch.nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = torch.nn.Conv2d(
                input_channels,
                output_channels,
                shape,
                stride=stride,
                padding=shape // 2,
            )
            self.bn_3 = torch.nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class Encoder(torch.nn.Module):
    """Short-chunk CNN architecture with residual connections.

    Args:
        sample_rate (float): Audio input sampling rate.
        n_channels (int): Number of convolutional channels. Default: 128
        n_fft (int): FFT size for computing melspectrogram. Default: 1024
        n_mels (int): Number of mel frequency bins: Default 128

    Adapted from https://github.com/minzwon/sota-music-tagging-models. Licensed under MIT by Minz Won.
    """

    def __init__(
        self,
        d_embed: int,
        sample_rate: float,
        n_channels: int = 128,
        n_fft: int = 1024,
        n_mels: int = 128,
    ):
        super().__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            n_mels=n_mels,
        )
        # self.to_db = torchaudio.transforms.AmplitudeToDB()
        # self.spec_bn = torch.nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer7 = Res_2d(n_channels * 2, n_channels * 4, stride=2)

        # Dense
        self.dense1 = torch.nn.Linear(n_channels * 4, d_embed)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        """Compute aggregated embeddings for batch of waveform tensors.

        Args:
            x (torch.Tensor): Batch of waveform tensors with shape (bs, 1, seq_len).

        Returns:
            z (torch.Tensor): Embeddings of shape (bs, d_embed).
        """
        # Spectrogram
        x = self.spec(x)
        # x = self.to_db(x)
        x = torch.pow(x.abs() + 1e-8, 0.3)
        x = x.unsqueeze(1)
        # x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = torch.nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.relu(x)

        return x


class PostProcessor(torch.nn.Module):
    def __init__(self, num_params: int, d_embed: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_embed, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, num_params),
            torch.nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        return self.mlp(z)


class Mixer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_gain_dB: int = -48.0,
        max_gain_dB: int = 48.0,
    ) -> None:
        super().__init__()
        self.num_params = 2
        self.param_names = ["Gain dB", "Pan"]
        self.sample_rate = sample_rate
        self.min_gain_dB = min_gain_dB
        self.max_gain_dB = max_gain_dB

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        """Generate a mix of stems given mixing parameters normalized to (0,1).

        Args:
            x (torch.Tensor): Batch of waveform stem tensors with shape (bs, num_tracks, seq_len).
            p (torch.Tensor): Batch of normalized mixing parameters (0,1) for each stem with shape (bs, num_tracks, num_params)

        Returns:
            y (torch.Tensor): Batch of stereo waveform mixes with shape (bs, 2, seq_len)
        """
        bs, num_tracks, seq_len = x.size()

        # ------------- apply gain -------------
        gain_dB = p[..., 0]  # get gain parameter
        gain_dB = restore_from_0to1(gain_dB, self.min_gain_dB, self.max_gain_dB)
        gain_lin = 10 ** (gain_dB / 20.0)  # convert gain from dB scale to linear
        gain_lin = gain_lin.view(bs, num_tracks, 1)  # reshape for multiplication
        x = x * gain_lin  # apply gain (bs, num_tracks, seq_len)

        # ------------- apply panning -------------
        # expand mono stems to stereo, then apply panning
        x = x.view(bs, num_tracks, 1, -1)  # (bs, num_tracks, 1, seq_len)
        x = x.repeat(1, 1, 2, 1)  # (bs, num_tracks, 2, seq_len)

        pan = p[..., 1]  # get pan parameter
        pan_theta = pan * torch.pi / 2
        left_gain = torch.cos(pan_theta)
        right_gain = torch.sin(pan_theta)
        pan_gains_lin = torch.stack([left_gain, right_gain], dim=-1)
        pan_gains_lin = pan_gains_lin.view(bs, num_tracks, 2, 1)  # reshape for multiply
        x = x * pan_gains_lin  # (bs, num_tracks, 2, seq_len)

        # ----------------- apply mix -------------
        # generate a mix for each batch item by summing stereo tracks
        y = torch.sum(x, dim=1)  # (bs, 2, seq_len)

        p = torch.cat(
            (
                gain_dB.view(bs, num_tracks, 1),
                pan.view(bs, num_tracks, 1),
            ),
            dim=-1,
        )

        return y, p


class DifferentiableMixingConsole(torch.nn.Module):
    """Differentiable mixing console.

    Notes:
        We do not use neural audio effect proxies as in the original publication.
        Instead we use a set of explicitly differentiable audio effects.

    Steinmetz et al. (2021). Automatic multitrack mixing with a differentiable mixing console of neural audio effects. ICASSP.
    """

    def __init__(self, sample_rate: int, d_embed: int = 128) -> None:
        super().__init__()

        # Creates a mix given tracks and parameters (also called the "Transformation Network")
        self.mixer = Mixer(sample_rate)

        # Simple 2D CNN on spectrograms
        self.encoder = Encoder(d_embed, sample_rate)

        # MLP projects embedding + context to parameter space
        self.post_processor = PostProcessor(self.mixer.num_params, d_embed * 2)

    def forward(self, x: torch.Tensor):
        """Given a set of tracks, analyze them with a shared encoder, predict a set of mixing parameters,
        and use these parameters to generate a stereo mixture of the inputs.

        Args:
            x (torch.Tensor): Input tracks with shape (bs, num_tracks, seq_len)

        Returns:
            y (torch.Tensor): Final stereo mixture with shape (bs, 2, seq_len)
            p (torch.Tensor): Estimated (denormalized) mixing parameters with shape (bs, num_tracks, num_params)
        """
        bs, num_tracks, seq_len = x.size()

        # move tracks to the batch dimension to fully parallelize embedding computation
        x = x.view(bs * num_tracks, -1)

        # generate single embedding for each track
        e = self.encoder(x)
        e = e.view(bs, num_tracks, -1)  # (bs, num_tracks, d_embed)

        # generate the "context" embedding
        c = e.mean(dim=1, keepdim=True)  # (bs, 1, d_embed)
        c = c.repeat(1, num_tracks, 1)  # (bs, num_tracks, d_embed)

        # fuse the track emb and context emb
        ec = torch.cat((e, c), dim=-1)  # (bs, num_tracks, d_embed*2)

        # estimate mixing parameters for each track (in parallel)
        p = self.post_processor(ec)  # (bs, num_tracks, num_params)

        # generate the stereo mix
        x = x.view(bs, num_tracks, -1)  # move tracks back from batch dim
        y, p = self.mixer(x, p)  # (bs, 2, seq_len) # and denormalized params

        return y, p
