import os
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger


from automix.system import System
from automix.callbacks import LogAudioCallback
from automix.data import ENSTDrumsDataset
from automix.callbacks import LogAudioCallback


if __name__ == "__main__":
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--train_length", type=int, default=262144)
    parser.add_argument("--val_length", type=int, default=262144)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser = System.add_model_specific_args(parser)  # add model specific args
    parser = pl.Trainer.add_argparse_args(parser)  # add all Trainer options
    args = parser.parse_args()  # parse them args

    # setup callbacks
    callbacks = [
        LogAudioCallback(),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            filename=f"{args.dataset_name}-{args.automix_model}"
            + "_epoch-{epoch}-step-{step}",
            monitor="val/loss_epoch",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    wandb_logger = WandbLogger(save_dir=args.log_dir)

    # create PyTorch Lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=callbacks
    )

    # create the System
    system = System(**vars(args))

    train_dataset = ENSTDrumsDataset(
        args.dataset_dir,
        args.train_length,
        44100,
        drummers=[1, 2],
        num_examples_per_epoch=10000,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    val_dataset = ENSTDrumsDataset(
        args.dataset_dir,
        args.train_length,
        44100,
        drummers=[3],
        num_examples_per_epoch=1000,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # train!
    trainer.fit(system, train_dataloader, val_dataloader)
