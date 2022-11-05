import os
import glob
import torch
import argparse
import torchaudio
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional


class ENSTDrumsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        length: int,
        sample_rate: float,
        drummers: List[int] = [1, 2],
        track_names: List[str] = [
            "kick",
            "snare",
            "hi-hat",
            "overhead_L",
            "overhead_R",
            "tom_1",
            "tom_2",
        ],
        wet_mix: bool = False,
        hits: bool = False,
        num_examples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.length = length
        self.sample_rate = sample_rate
        self.drummers = drummers
        self.track_names = track_names
        self.wet_mix = wet_mix
        self.hits = hits
        self.num_examples_per_epoch = num_examples_per_epoch

        # find all mixes
        self.mix_filepaths = []
        for drummer in drummers:
            search_path = os.path.join(
                root_dir,
                f"drummer_{drummer}",
                "audio",
                "wet_mix" if wet_mix else "dry_mix",
                "*.wav",
            )
            self.mix_filepaths += glob.glob(search_path)

        # remove any mixes that are shorter than the requested length
        self.mix_filepaths = [
            fp
            for fp in self.mix_filepaths
            if torchaudio.info(fp).num_frames > self.length
        ]

        self.mix_filepaths = self.mix_filepaths[0:1]

        # remove any mixes that are just hits
        if not self.hits:
            self.mix_filepaths = [fp for fp in self.mix_filepaths if "hits" not in fp]

        if len(self.mix_filepaths) < 1:
            raise RuntimeError(f"No files found in {self.root_dir}.")
        else:
            print(f"Found {len(self.mix_filepaths)} examples from drummers: {drummers}")

        # for now we assume all songs have all tracks

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _):
        # select a mix at random
        mix_idx = np.random.randint(0, len(self.mix_filepaths))
        mix_filepath = self.mix_filepaths[mix_idx]
        example_id = os.path.basename(mix_filepath)
        drummer_id = os.path.normpath(mix_filepath).split(os.path.sep)[-4]

        # get random offset
        md = torchaudio.info(mix_filepath)  # check length
        offset = np.random.randint(0, md.num_frames - self.length - 1)
        offset = 524288

        # load the chunk of the mix
        y, sr = torchaudio.load(
            mix_filepath,
            frame_offset=offset,
            num_frames=self.length,
        )
        y *= 0.5  # reduce gain of the target mix

        # get all the tracks
        x = []
        for track_name in self.track_names:
            track_filepath = os.path.join(
                self.root_dir,
                drummer_id,
                "audio",
                track_name,
                example_id,
            )
            x_s, sr = torchaudio.load(
                track_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )

            # apply random gain augmentation (-12 to +12 dB)
            gain_dB = (torch.rand(1) * 24) - 12
            gain_lin = 10 ** (gain_dB / 20.0)
            x_s *= gain_lin
            x.append(x_s)

        x = torch.cat(x)  # collect the tracks into tensor

        return x, y


class ENSTDrumsDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # download
        return

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset_train = ENSTDrumsDataset(
                self.hparams.data_dir,
                self.hparams.length,
                self.hparams.sample_rate,
                drummers=self.hparams.drummers,
                wet_mix=self.hparams.wet_mix,
                num_examples_per_epoch=self.hparams.num_example_per_training_epoch,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_train = ENSTDrumsDataset(
                self.hparams.data_dir,
                self.hparams.length,
                self.hparams.sample_rate,
                drummers=self.hparams.drummers,
                wet_mix=self.hparams.wet_mix,
                num_examples_per_epoch=self.hparams.num_example_per_testing_epoch,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--num_examples_per_training_epoch", type=int, default=1000)
        parser.add_argument(
            "--num_examples_per_validation_epoch", type=int, default=100
        )
        parser.add_argument("--num_examples_per_testing_epoch", type=int, default=100)
        return parser
