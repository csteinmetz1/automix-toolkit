import os
import csv
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger


from automix.system import System
from automix.callbacks import LogAudioCallback
from automix.data import ENSTDrumsDataset, MedleyDBDataset, DSD100Dataset
from automix.callbacks import LogAudioCallback


if __name__ == "__main__":
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--dataset_dirs", nargs="+")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--train_length", type=int, default=262144)
    parser.add_argument("--val_length", type=int, default=262144)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_num_tracks", type=int, default=8)
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

    wandb_logger = WandbLogger(save_dir=args.log_dir, project="automix")

    # create PyTorch Lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=callbacks
    )

    # create the System
    system = System(**vars(args))

    if "ENST-drums" in args.dataset_name:
        wet_mix = True if "wet" in args.dataset_name else False
        train_dataset = ENSTDrumsDataset(
            args.dataset_dirs[0],
            args.train_length,
            44100,
            drummers=[1, 2, 3],
            indices=[0, 168],
            num_examples_per_epoch=1000,
            wet_mix=wet_mix,
        )
        val_dataset = ENSTDrumsDataset(
            args.dataset_dirs[0],
            args.train_length,
            44100,
            drummers=[1, 2, 3],
            indices=[168, 189],
            num_examples_per_epoch=100,
            wet_mix=wet_mix,
        )
        test_dataset = ENSTDrumsDataset(
            args.dataset_dirs[0],
            args.train_length,
            44100,
            drummers=[1, 2, 3],
            indices=[189, 210],
            num_examples_per_epoch=100,
            wet_mix=wet_mix,
        )
        train_set = train_dataset.mix_filepaths
        val_set = val_dataset.mix_filepaths
        test_set = test_dataset.mix_filepaths
    elif args.dataset_name == "DSD100":
        train_dataset = DSD100Dataset(
            args.dataset_dirs[0],
            args.train_length,
            44100,
            indices=[0, 80],
            num_examples_per_epoch=1000,
        )
        val_dataset = DSD100Dataset(
            args.dataset_dirs[0],
            args.train_length,
            44100,
            indices=[80, 90],
            num_examples_per_epoch=100,
        )
        test_dataset = DSD100Dataset(
            args.dataset_dirs[0],
            args.train_length,
            44100,
            indices=[90, 100],
            num_examples_per_epoch=100,
        )
        train_set = train_dataset.mix_dirs
        val_set = val_dataset.mix_dirs
        test_set = test_dataset.mix_dirs
    elif args.dataset_name == "MedleyDB":
        train_dataset = MedleyDBDataset(
            args.dataset_dirs,
            args.train_length,
            44100,
            indices=[0, 156],
            max_num_tracks=args.max_num_tracks,
            num_examples_per_epoch=10000,
        )
        val_dataset = MedleyDBDataset(
            args.dataset_dirs,
            args.val_length,
            44100,
            indices=[156, 175],
            max_num_tracks=args.max_num_tracks,
            num_examples_per_epoch=100,
        )
        test_dataset = MedleyDBDataset(
            args.dataset_dirs,
            args.val_length,
            44100,
            indices=[175, 197],
            max_num_tracks=args.max_num_tracks,
            num_examples_per_epoch=100,
        )
        train_set = train_dataset.mix_dirs
        val_set = val_dataset.mix_dirs
        test_set = test_dataset.mix_dirs

    # create a file with the train/val/test split
    csv_filepath = os.path.join(wandb_logger.experiment.dir, "split.csv")
    print(csv_filepath)
    with open(csv_filepath, "w") as f:
        writer = csv.writer(f)
        for fp in train_set:
            writer.writerow(["train", fp])
        for fp in val_set:
            writer.writerow(["val", fp])
        for fp in test_set:
            writer.writerow(["test", fp])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
    )

    # train!
    trainer.fit(system, train_dataloader, val_dataloader)
