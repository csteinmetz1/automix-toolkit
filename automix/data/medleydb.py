import os
import glob
import torch
import random
import torchaudio
import numpy as np

from tqdm import tqdm
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
        buffer_size_gb: float = 3.0,
        buffer_reload_rate: int = 1000,
        normalization: str = "peak",
    ) -> None:
        super().__init__()
        self.root_dirs = root_dirs
        self.length = length
        self.sample_rate = sample_rate
        self.indices = indices
        self.max_num_tracks = max_num_tracks
        self.num_examples_per_epoch = num_examples_per_epoch
        self.buffer_size_gb = buffer_size_gb
        self.buffer_reload_rate = buffer_reload_rate
        self.normalization = normalization

        self.mix_dirs = []
        for root_dir in root_dirs:
            # find all mix directories
            mix_dirs = glob.glob(os.path.join(root_dir, "*"))
            # remove items that are not directories
            mix_dirs = [mix_dir for mix_dir in mix_dirs if os.path.isdir(mix_dir)]
            mix_dirs = sorted(mix_dirs)  # sort
            self.mix_dirs += mix_dirs

        self.mix_dirs = self.mix_dirs[indices[0] : indices[1]]  # select subset
        self.items_since_load = self.buffer_reload_rate

    def reload_buffer(self):

        self.examples = []  # clear buffer
        self.items_since_load = 0  # reset iteration counter
        nbytes_loaded = 0  # counter for data in RAM

        # different subset in each
        random.shuffle(self.mix_dirs)

        # load files into RAM
        for mix_dir in self.mix_dirs:
            mix_id = os.path.basename(mix_dir)
            mix_filepath = glob.glob(os.path.join(mix_dir, "*.wav"))[0]

            # now check the length of the mix
            try:
                y, sr = torchaudio.load(mix_filepath)
            except:
                print(f"Skipping {mix_filepath}")
                continue

            mix_num_frames = y.shape[-1]
            nbytes = y.element_size() * y.nelement()
            nbytes_loaded += nbytes

            # now find all the track filepaths
            track_filepaths = glob.glob(os.path.join(mix_dir, f"{mix_id}_RAW", "*.wav"))

            # check length of each track
            tracks = []
            for tidx, track_filepath in enumerate(track_filepaths):
                x, sr = torchaudio.load(track_filepath)
                tracks.append(x)

                nbytes = x.element_size() * x.nelement()
                nbytes_loaded += nbytes

                track_num_frames = x.shape[-1]
                if track_num_frames < mix_num_frames:
                    mix_num_frames = track_num_frames

            # store this example
            example = {
                "mix_id": os.path.dirname(mix_filepath).split(os.sep)[-1],
                "mix_filepath": mix_filepath,
                "mix_audio": y,
                "num_frames": mix_num_frames,
                "track_filepaths": track_filepaths,
                "track_audio": tracks,
            }

            self.examples.append(example)

            # check the size of loaded data
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _):

        # increment counter
        self.items_since_load += 1

        # load next chunk into buffer if needed
        if self.items_since_load > self.buffer_reload_rate:
            self.reload_buffer()

        # select an example at random
        example_idx = np.random.randint(0, len(self.examples))
        example = self.examples[example_idx]

        silent = True
        counter = 0
        while silent:
            # get the whole mix
            y = example["mix_audio"]

            # get random offset
            start_idx = np.random.randint(0, example["num_frames"] - self.length - 1)
            end_idx = start_idx + self.length
            y = y[:, start_idx:end_idx]

            energy = (y**2).mean()

            if y.shape[-1] == self.length and energy > 1e-4:
                silent = False

            counter += 1

            y /= y.abs().max()

        # -------------------- load the tracks from disk --------------------
        x = torch.zeros((self.max_num_tracks, self.length))
        pad = [True] * self.max_num_tracks  # note which tracks are empty
        random.shuffle(example["track_audio"])  # load random tracks each time

        tidx = 0
        for track in example["track_audio"]:
            x_s = track[:, start_idx:end_idx]

            energy = (x_s**2).mean()
            if energy > 1e-6:  # ensure track is active
                gain_dB = -12
                gain_lin = 10 ** (gain_dB / 20.0)
                x_s /= x_s.abs().max().clamp(1e-8)
                x_s *= gain_lin

                x[tidx, :] = x_s
                pad[tidx] = False
                tidx += 1

            if (tidx + 1) >= self.max_num_tracks:
                break

        return x, y, pad
