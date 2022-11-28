import torch
import torchaudio

from automix.utils import restore_from_0to1


class VGGishEncoder(torch.nn.Module):
    def __init__(self, sample_rate: float) -> None:
        super().__init__()
        model = torch.hub.load("harritaylor/torchvggish", "vggish")
        model.eval()
        self.sample_rate = sample_rate
        self.model = model
        self.d_embed = 128
        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        bs, seq_len = x.size()
        with torch.no_grad():
            if self.sample_rate != 16000:
                x = self.resample(x)
            z = []
            for bidx in range(bs):
                x_item = x[bidx : bidx + 1, :]
                x_item = x_item.permute(1, 0)
                x_item = x_item.cpu().view(-1).numpy()
                z_item = self.model(x_item, fs=16000)
                z_item = z_item.mean(dim=0)  # mean across time frames
                z.append(z_item)
            z = torch.cat(z, dim=0)
        return z


class Res_2d(torch.nn.Module):
    """Residual 2D Convolutional layer.

    Args:
        input_channels (int):

    Adapted from https://github.com/minzwon/sota-music-tagging-models. Licensed under MIT by Minz Won.
    """

    def __init__(self, input_channels: int, output_channels, shape=3, stride=2):
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


class ShortChunkCNN_Res(torch.nn.Module):
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
        sample_rate,
        n_channels=128,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        ckpt_path: str = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = torch.nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer7 = Res_2d(n_channels * 2, n_channels * 4, stride=2)

        # Dense
        self.dense1 = torch.nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = torch.nn.BatchNorm1d(n_channels * 4)
        self.dense2 = torch.nn.Linear(n_channels * 4, n_class)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Loaded weights from {ckpt_path}")

        self.d_embed = n_channels * 4
        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)

    def forward(self, x):

        # resampling
        if self.sample_rate != 16000:
            x = self.resample(x)

        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

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
        # x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.dense2(x)
        # x = nn.Sigmoid()(x)

        return x


class PostProcessor(torch.nn.Module):
    def __init__(self, num_params: int, d_embed: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_embed, 256),
            torch.nn.Dropout(0.2),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Dropout(0.2),
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
        max_gain_dB: int = 24.0,
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

    def __init__(
        self,
        sample_rate: int,
        encoder_arch: str = "short_res",
        load_weights: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder_arch = encoder_arch

        # Creates a mix given tracks and parameters (also called the "Transformation Network")
        self.mixer = Mixer(sample_rate)

        # Simple 2D CNN on spectrograms
        if encoder_arch == "vggish":
            self.encoder = VGGishEncoder(sample_rate)
        elif encoder_arch == "short_res":
            self.encoder = ShortChunkCNN_Res(
                sample_rate,
                ckpt_path="./checkpoints/encoder.ckpt" if load_weights else None,
            )
        else:
            raise ValueError(f"Invalid encoder_arch: {encoder_arch}")

        # MLP projects embedding + context to parameter space
        self.post_processor = PostProcessor(
            self.mixer.num_params,
            self.encoder.d_embed * 2,
        )

    def block_based_forward(
        self,
        x: torch.Tensor,
        block_size: int,
        hop_length: int,
    ):
        bs, num_tracks, seq_len = x.size()

        x = torch.nn.functional.pad(
            x,
            (block_size // 2, block_size // 2),
            mode="reflect",
        )

        unfold_fn = torch.nn.Unfold(
            (1, block_size),
            stride=(1, hop_length),
        )
        fold_fn = torch.nn.Fold(
            (1, x.shape[-1]),
            (1, block_size),
            stride=(1, hop_length),
        )
        window = torch.hann_window(block_size)
        window = window.view(1, 1, -1)
        window = window.type_as(x)

        x_track_blocks = []
        for track_idx in range(num_tracks):
            x_track_blocks.append(unfold_fn(x[:, track_idx, :].view(bs, 1, 1, -1)))

        x_blocks = torch.stack(x_track_blocks, dim=1)
        num_blocks = x_blocks.shape[-1]

        block_outputs = []
        for block_idx in range(num_blocks):
            x_block = x_blocks[..., block_idx]
            x_result, _ = self.forward(x_block)
            x_result = x_result.view(bs, 1, 2, -1)
            block_outputs.append(x_result)

        block_outputs = torch.cat(block_outputs, dim=1)
        block_outputs = block_outputs * window  # apply overlap-add window
        y_left = fold_fn(block_outputs[:, :, 0, :].permute(0, 2, 1))
        y_right = fold_fn(block_outputs[:, :, 1, :].permute(0, 2, 1))
        y = torch.cat((y_left, y_right), dim=1)

        # crop the padded areas
        y = y[..., block_size // 2 : -(block_size // 2)]

        return y

    def forward(self, x: torch.Tensor, track_mask: torch.Tensor = None):
        """Given a set of tracks, analyze them with a shared encoder, predict a set of mixing parameters,
        and use these parameters to generate a stereo mixture of the inputs.

        Args:
            x (torch.Tensor): Input tracks with shape (bs, num_tracks, seq_len)
            track_mask (torch.Tensor, optional): Mask specifying inactivate tracks with shape (bs, num_tracks)

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
        c = []
        for bidx in range(bs):
            c_n = e[bidx, ~track_mask[bidx, :], :].mean(
                dim=0, keepdim=True
            )  # (bs, 1, d_embed)
            c_n = c_n.repeat(num_tracks, 1)  # (bs, num_tracks, d_embed)
            c.append(c_n)
        c = torch.stack(c, dim=0)

        # fuse the track embs and context embs
        ec = torch.cat((e, c), dim=-1)  # (bs, num_tracks, d_embed*2)

        # estimate mixing parameters for each track (in parallel)
        p = self.post_processor(ec)  # (bs, num_tracks, num_params)

        # generate the stereo mix
        x = x.view(bs, num_tracks, -1)  # move tracks back from batch dim
        y, p = self.mixer(x, p)  # (bs, 2, seq_len) # and denormalized params

        return y, p
