import os
import torch
import auraloss
import pytorch_lightning as pl
from argparse import ArgumentParser

from automix.models.dmc import DifferentiableMixingConsole
from automix.models.mixwaveunet import MixWaveUNet


class System(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False  # Enable manual optimization

        # create the model
        if self.hparams.automix_model == "mixwaveunet":
            self.model = MixWaveUNet(self.hparams.max_num_tracks, 2)
        elif self.hparams.automix_model == "dmc":
            self.model = DifferentiableMixingConsole(
                self.hparams.sample_rate,
                load_weights=self.hparams.pretrained_encoder,
            )
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
            elif recon_loss == "sisdr":
                self.recon_losses[recon_loss] = auraloss.time.SISDRLoss()
            elif recon_loss == "sd":
                self.recon_losses[recon_loss] = auraloss.freq.SumAndDifferenceSTFTLoss(
                    fft_sizes=[4096, 1024, 256],
                    hop_sizes=[2048, 512, 128],
                    win_lengths=[4096, 1024, 256],
                )
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

    def forward(self, x: torch.Tensor, track_mask: torch.Tensor = None) -> torch.Tensor:
        """Apply model to audio waveform tracks.
        Args:
            x (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
        Returns:
            y_hat (torch.Tensor): Stereo mix with shape (bs, 2, seq_len)
        """
        return self.model(x, track_mask)

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        train: bool = False,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing tracks (x) mix audio (y).
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
            train (bool): Wether step is called during training (True) or validation (False).
        """
        x, y, track_mask = batch  # tracks, mix, mask

        # process input audio with model
        y_hat, params = self(x, track_mask)

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
            "p": params.detach().cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, train=True)

        # Manually perform optimization step
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

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

        if self.hparams.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
        elif self.hparams.schedule == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                [
                    int(self.hparams.max_epochs * 0.85),
                    int(self.hparams.max_epochs * 0.95),
                ],
            )
        else:
            return optimizer

        lr_schedulers = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        return [optimizer], lr_schedulers

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        # --- Loss functions  ---
        parser.add_argument("--recon_losses", nargs="+", default=["sd"])
        parser.add_argument("--recon_loss_weights", nargs="+", default=[1.0])
        parser.add_argument("--schedule", type=str, default="cosine")
        # --- Model ---
        parser.add_argument("--automix_model", type=str, default="dmc")
        parser.add_argument("--pretrained_encoder", action="store_true")

        return parser
