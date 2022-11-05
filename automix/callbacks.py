import io
import wandb
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


def build_param_table(params, param_names):
    """

    Args:
        params (torch.Tensor): Parameters for one mix with shape (num_tracks, num_params)

    """
    num_tracks, num_params = params.size()

    table = "<html><table>\n"
    table += "<tr><th>Track</th>\n"

    for param_name in param_names:
        table += f"<th>{param_name}</th>\n"

    table += "</tr>\n"

    for track_idx in range(num_tracks):
        table += f"<tr><td>{track_idx}</td>\n"
        for param_idx in range(num_params):
            table += f"<td>{params[track_idx, param_idx]:0.2f}</td>\n"

        table += "\n</tr>\n"

    table += "</table>\n</html>"

    return table


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
        sample_rate = pl_module.hparams.sample_rate

        if outputs is not None:
            num_examples = outputs["x"].shape[0]
            if num_examples > self.num_examples:
                num_examples = self.num_examples

            if batch_idx == 0:

                columns = ["input", "output", "target", "params"]
                data = []

                for elem_idx in range(num_examples):
                    x = outputs["x"][elem_idx, ...].float()
                    # sum the input stems to create mono mix
                    x = x.sum(dim=0, keepdim=True)
                    x /= x.abs().max()

                    y = outputs["y"][elem_idx, ...].float()
                    y /= y.abs().max()

                    y_hat = outputs["y_hat"][elem_idx, ...].float()
                    y_hat /= y_hat.abs().max()

                    entries = [
                        wandb.Audio(x.T, sample_rate, caption=f"{elem_idx}"),
                        wandb.Audio(y_hat.T, sample_rate, caption=f"{elem_idx}"),
                        wandb.Audio(y.T, sample_rate, caption=f"{elem_idx}"),
                    ]

                    if pl_module.hparams.automix_model == "dmc":
                        p = outputs["p"][elem_idx, ...].float()
                        param_table = build_param_table(
                            p, pl_module.model.mixer.param_names
                        )
                        entries.append(wandb.Html(param_table))

                    data.append(entries)

                trainer.logger.log_table(key="audio", columns=columns, data=data)
