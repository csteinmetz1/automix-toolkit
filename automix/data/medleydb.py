import os
import glob
import torch
import random
import torchaudio
import numpy as np

from typing import List


class MedleyDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs: List[str],
        length: int,
        sample_rate: float,
        indices: List[int] = [0, 100],
        max_num_tracks: int = 16,
        num_examples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.root_dirs = root_dirs
        self.length = length
        self.sample_rate = sample_rate
        self.indices = indices
        self.max_num_tracks = max_num_tracks
        self.num_examples_per_epoch = num_examples_per_epoch

        self.mix_dirs = []
        for root_dir in root_dirs:
            # find all mix directories
            mix_dirs = glob.glob(os.path.join(root_dir, "*"))
            # remove items that are not directories
            mix_dirs = [mix_dir for mix_dir in mix_dirs if os.path.isdir(mix_dir)]
            mix_dirs = sorted(mix_dirs)  # sort
            self.mix_dirs += mix_dirs

        self.mix_dirs = self.mix_dirs[indices[0] : indices[1]]  # select subset

        self.examples = []
        # check for mix and tracks in each directory
        for mix_dir in self.mix_dirs:
            mix_id = os.path.basename(mix_dir)
            mix_filepath = glob.glob(os.path.join(mix_dir, "*.wav"))[0]

            # now check the length of the mix
            try:
                mix_num_frames = torchaudio.info(mix_filepath).num_frames
            except:
                print(f"Skipping {mix_filepath}")
                continue

            # now find all the track filepaths
            track_filepaths = glob.glob(os.path.join(mix_dir, f"{mix_id}_RAW", "*.wav"))

            # check length of each track
            for tidx, track_filepath in enumerate(track_filepaths):
                track_num_frames = torchaudio.info(track_filepath).num_frames

            # store this example
            example = {
                "mix_id": os.path.dirname(mix_filepath).split(os.sep)[-1],
                "mix_filepath": mix_filepath,
                "num_frames": mix_num_frames,
                "track_filepaths": track_filepaths,
            }
            self.examples.append(example)

        if len(self.examples) < 1:
            raise RuntimeError(f"No examples found in {self.root_dir}.")
        else:
            print(
                f"Found {len(self.examples)} examples from {indices[0]} to {indices[1]}"
            )

        # for now we assume all songs have all tracks

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _):
        # select an example at random
        example_idx = np.random.randint(0, len(self.examples))
        example = self.examples[example_idx]

        # get random offset
        offset = np.random.randint(0, example["num_frames"] - self.length - 1)

        # load the chunk of the mix
        y, sr = torchaudio.load(
            example["mix_filepath"],
            frame_offset=offset,
            num_frames=self.length,
        )
        print(y.shape, example["mix_filepath"], example["num_frames"])
        y /= y.abs().max()

        # -------------------- load the tracks from disk --------------------
        x = torch.zeros((self.max_num_tracks, self.length))
        pad = [True] * self.max_num_tracks  # note which tracks are empty
        random.shuffle(example["track_filepaths"])  # load random tracks each time

        for tidx, track_filepath in enumerate(example["track_filepaths"]):
            x_s, sr = torchaudio.load(
                track_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )

            # apply random gain augmentation (-12 to +12 dB)
            gain_dB = (torch.rand(1) * 24) - 12
            gain_lin = 10 ** (gain_dB / 20.0)
            x_s *= gain_lin

            x[tidx, :] = x_s
            pad[tidx] = False

            if (tidx + 1) >= self.max_num_tracks:
                break

        return x, y, pad
