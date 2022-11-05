import os
import glob
import torch
import random
import torchaudio
import numpy as np

from typing import List


class MixingSecretsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        length: int,
        sample_rate: float,
        indices: List[int] = [0, 100],
        max_num_tracks: int = 16,
        num_examples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.length = length
        self.sample_rate = sample_rate
        self.indices = indices
        self.max_num_tracks = max_num_tracks
        self.num_examples_per_epoch = num_examples_per_epoch

        # find all mix directories
        mix_dirs = glob.glob(os.path.join(root_dir, "*"))
        # remove items that are not directories
        self.mix_dirs = [mix_dir for mix_dir in mix_dirs if os.path.isdir(mix_dir)]
        self.mix_dirs = sorted(self.mix_dirs)  # sort
        self.mix_dirs = self.mix_dirs[indices[0] : indices[1]]  # select subset

        self.examples = []
        # check for mix and tracks in each directory
        for mix_dir in self.mix_dirs:

            mix_id = os.path.basename(mix_dir)

            mix_filepaths = glob.glob(os.path.join(mix_dir, "mixes", "*"))
            # skip this song if no mixes found
            if len(mix_filepaths) < 1:
                continue
            # if there is more than one mix, use the one with "Full" in name
            elif len(mix_filepaths) > 1:
                mix_filepaths = [mix_fp for mix_fp in mix_filepaths if "Full" in mix_fp]
                if len(mix_filepaths) == 0:
                    continue
                else:
                    mix_filepath = mix_filepaths[0]
            # if only one mix use that
            else:
                mix_filepath = mix_filepaths[0]

            # now check the length of the mix
            mix_num_frames = torchaudio.info(mix_filepath).num_frames

            # now find all the track filepaths
            track_filepaths = glob.glob(os.path.join(mix_dir, "tracks", "*.wav"))

            # check length of each track
            for tidx, track_filepath in enumerate(track_filepaths):
                track_num_frames = torchaudio.info(track_filepath).num_frames
                print(
                    tidx, mix_filepath, track_filepath, track_num_frames, mix_num_frames
                )

            # store this example
            example = {
                "mix_id": os.path.dirname(mix_filepath).split(os.sep)[-2],
                "mix_filepath": mix_filepath,
                "num_frames": num_frames,
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
        y *= 0.5  # reduce gain of the target mix

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

            # store
            print(
                example["mix_id"],
                x.shape,
                y.shape,
                x_s.shape,
                offset,
                example["num_frames"],
            )
            x[tidx, :] = x_s
            pad[tidx] = False

            if (tidx + 1) > self.max_num_tracks:
                break

        return x, y, pad
