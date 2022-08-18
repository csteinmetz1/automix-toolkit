import os
import torch
import auraloss
import pytorch_lightning as pl
from argparse import ArgumentParser

from automix.models.dmc import DifferentiableMixingConsole
from automix.models.mixwaveunet import MixWaveUNet
from automix.models.mixwaveunet.simple_waveunet import SimpleWaveUNet
from automix.utils import center_crop


class System(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        # create the model
        if self.hparams.automix_model == "mixwaveunet":
            self.model = MixWaveUNet(7, self.hparams.train_length)
        elif self.hparams.automix_model == "simple-waveunet":
            self.model = SimpleWaveUNet(7, 2)
        elif self.hparams.automix_model == "dmc":
            self.model = DifferentiableMixingConsole()
        else:
            raise RuntimeError(f"Invalid automix_model: {self.hparams.automix_model}")

        self.recon_losses = torch.nn.ModuleDict()
        for recon_loss in self.hparams.recon_losses:
            if recon_loss == "mrstft":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 2048, 8192],
                    hop_sizes=[256, 1024, 4096],
                    win_lengths=[512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "l1":
                self.recon_losses[recon_loss] = torch.nn.L1Loss()
            else:
                raise RuntimeError(f"Invalid reconstruction loss: {recon_loss}")

        self.sisdr = auraloss.time.SISDRLoss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 2048, 8192],
            hop_sizes=[256, 1024, 4096],
            win_lengths=[512, 2048, 8192],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model to audio waveform tracks.
        Args:
            x (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
        Returns:
            y_hat (torch.Tensor): Stereo mix with shape (bs, 2, seq_len)
        """
        return self.model(x)

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        optimizer_idx: int = 0,
        train: bool = False,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing tracks (x) mix audio (y).
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
            train (bool): Wether step is called during training (True) or validation (False).
        """
        x, y = batch  # tracks, mix

        # process input audio with model
        y_hat = self(x)

        # compute loss
        loss = 0

        # compute loss on the waveform
        for loss_idx, (loss_name, loss_fn) in enumerate(self.recon_losses.items()):
            recon_loss = loss_fn(y_hat, y)
            loss += self.hparams.recon_loss_weights[loss_idx] * recon_loss

            self.log(
                ("train" if train else "val") + f"/{loss_name}",
                recon_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        sisdr_error = -self.sisdr(y_hat, y)
        # log the SI-SDR error
        self.log(
            ("train" if train else "val") + "/si-sdr",
            sisdr_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        mrstft_error = self.mrstft(y_hat, y)
        # log the MR-STFT error
        self.log(
            ("train" if train else "val") + "/mrstft",
            mrstft_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # log the overall loss
        self.log(
            ("train" if train else "val") + "/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # for plotting down the line
        data_dict = {
            "x": x.detach().float().cpu(),
            "y": y.detach().float().cpu(),
            "y_hat": y_hat.detach().float().cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, _ = self.common_step(batch, batch_idx, train=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=False)

        if batch_idx == 0:
            return data_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        lr_schedulers = {
            "scheduler": scheduler,
        }

        return [optimizer], lr_schedulers

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        # --- Loss functions  ---
        parser.add_argument("--recon_losses", nargs="+", default=["l1", "mrstft"])
        parser.add_argument("--recon_loss_weights", nargs="+", default=[100.0, 1.0])
        # --- Model ---
        parser.add_argument("--automix_model", type=str, default="mixwaveunet")

        return parser
