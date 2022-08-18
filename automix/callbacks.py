import io
import torch
import librosa
import PIL.Image
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from typing import Any
from torchvision.transforms import ToTensor


def fig2img(fig, dpi=120):
    """Convert a matplotlib figure to JPEG to be show in Tensorboard."""
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image


def plot_spectrograms(
    input: torch.Tensor,
    target: torch.Tensor,
    estimate: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    sample_rate: float = 48000,
    filename: Any = None,
):
    """Create a side-by-side plot of the attention weights and the spectrogram.
    Args:
        input (torch.Tensor): Input audio tensor with shape [1 x samples].
        target (torch.Tensor): Target audio tensor with shape [1 x samples].
        estimate (torch.Tensor): Estimate of the target audio with shape [1 x samples].
        n_fft (int, optional): Analysis FFT size.
        hop_length (int, optional): Analysis hop length.
        sample_rate (float, optional): Audio sample rate.
        filename (str, optional): If a filename is supplied, the plot is saved to disk.
    """
    # use librosa to take stft
    x_stft = librosa.stft(
        input.view(-1).numpy(),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    x_D = librosa.amplitude_to_db(
        np.abs(x_stft),
        ref=np.max,
    )

    y_stft = librosa.stft(
        target.view(-1).numpy(),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    y_D = librosa.amplitude_to_db(
        np.abs(y_stft),
        ref=np.max,
    )

    y_hat_stft = librosa.stft(
        estimate.view(-1).numpy(),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    y_hat_D = librosa.amplitude_to_db(
        np.abs(y_hat_stft),
        ref=np.max,
    )

    fig, axs = plt.subplots(
        nrows=3,
        sharex=True,
        figsize=(7, 6),
    )

    x_img = librosa.display.specshow(
        x_D,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_length,
        ax=axs[0],
    )

    y_img = librosa.display.specshow(
        y_D,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_length,
        ax=axs[1],
    )

    y_hat_img = librosa.display.specshow(
        y_hat_D,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_length,
        ax=axs[2],
    )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    return fig2img(fig)


class LogAudioCallback(pl.callbacks.Callback):
    def __init__(
        self,
        num_examples: int = 8,
        peak_normalize: bool = True,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.peak_normalize = peak_normalize
        self.sample_rate = sample_rate

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Called when the validation batch ends."""

        if outputs is not None:
            num_examples = outputs["x"].shape[0]
            if num_examples > self.num_examples:
                num_examples = self.num_examples

            if batch_idx == 0:
                for n in range(num_examples):
                    self.log_audio(
                        outputs,
                        n,
                        pl_module.hparams.sample_rate,
                        trainer.global_step,
                        trainer.logger,
                    )

    def log_audio(
        self,
        outputs,
        batch_idx: int,
        sample_rate: int,
        global_step: int,
        logger,
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        if "x" in outputs:
            x = outputs["x"][batch_idx, ...].float()
            x /= x.abs().max()

            # sum the input stems to create mono mix
            x = x.sum(dim=0, keepdim=True)

            logger.experiment.add_audio(
                f"{batch_idx+1}/input",
                x[0:1, :],
                global_step,
                sample_rate=sample_rate,
            )

        if "y" in outputs:
            y = outputs["y"][batch_idx, ...].float()
            y /= y.abs().max()

            logger.experiment.add_audio(
                f"{batch_idx+1}/target",
                y,
                global_step,
                sample_rate=sample_rate,
            )

        if "y_hat" in outputs:
            y_hat = outputs["y_hat"][batch_idx, ...].float()
            y_hat /= y_hat.abs().max()

            logger.experiment.add_audio(
                f"{batch_idx+1}/estimate",
                y_hat,
                global_step,
                sample_rate=sample_rate,
            )

        # skip spectrograms for now
        if "x" in outputs and "y" in outputs and "y_hat" in outputs and False:
            logger.experiment.add_image(
                f"spectrograms/{batch_idx+1}",
                plot_spectrograms(
                    x,
                    y,
                    y_hat,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    sample_rate=sample_rate,
                ),
                global_step,
            )
