import os
import glob
import torch
import random
import argparse
import torchaudio
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Tuple


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
            "tom_3",
        ],
        indices: Tuple[int, int] = [0, 1],
        wet_mix: bool = False,
        hits: bool = False,
        num_examples_per_epoch: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.length = length
        self.sample_rate = sample_rate
        self.drummers = drummers
        self.track_names = track_names
        self.indices = indices
        self.wet_mix = wet_mix
        self.hits = hits
        self.num_examples_per_epoch = num_examples_per_epoch
        self.max_num_tracks = 8  # this is fixed for this dataset

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

        # remove any mixes that have "norm" in the filename
        self.mix_filepaths = [fp for fp in self.mix_filepaths if not "norm" in fp]

        # self.mix_filepaths = self.mix_filepaths[0:1]

        # remove any mixes that are just hits
        if not self.hits:
            self.mix_filepaths = [fp for fp in self.mix_filepaths if "hits" not in fp]

        random.Random(seed).shuffle(self.mix_filepaths)
        self.mix_filepaths = self.mix_filepaths[indices[0] : indices[1]]

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

        md = torchaudio.info(mix_filepath)  # check length

        # load the chunk of the mix
        silent = True
        while silent:
            # get random offset
            offset = np.random.randint(0, md.num_frames - self.length - 1)

            y, sr = torchaudio.load(
                mix_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )
            energy = (y**2).mean()
            if energy > 1e-8:
                silent = False

        y /= y.abs().max().clamp(1e-8)  # peak normalize

        # -------------------- load the tracks from disk --------------------
        x = torch.zeros((self.max_num_tracks, self.length))
        pad = [True] * self.max_num_tracks  # note which tracks are empty

        for tidx, track_name in enumerate(self.track_names):
            track_filepath = os.path.join(
                self.root_dir,
                drummer_id,
                "audio",
                track_name,
                example_id,
            )
            if os.path.isfile(track_filepath):
                x_s, sr = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.length,
                )
                x_s /= x_s.abs().max().clamp(1e-6)
                x_s *= 10 ** (-12 / 20.0)
                x[tidx, :] = x_s
                pad[tidx] = False

        return x, y, pad
