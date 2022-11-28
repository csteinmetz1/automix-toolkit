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
        buffer_size_gb: float = 2.0,
        buffer_reload_rate: int = 4000,
        buffer_audio_length: int = 262144,
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
        self.buffer_audio_length = buffer_audio_length
        self.normalization = normalization

        self.mix_dirs = []
        for root_dir in root_dirs:
            # find all mix directories
            mix_dirs = glob.glob(os.path.join(root_dir, "*"))
            # remove items that are not directories
            mix_dirs = [mix_dir for mix_dir in mix_dirs if os.path.isdir(mix_dir)]
            mix_dirs = sorted(mix_dirs)  # sort
            self.mix_dirs += mix_dirs

        print(len(self.mix_dirs))
        for mix_dir in tqdm(self.mix_dirs):
            mix_id = os.path.basename(mix_dir)
            track_filepaths = glob.glob(os.path.join(mix_dir, f"{mix_id}_RAW", "*.wav"))
            # remove all mixes that have more tracks than requested
            if len(track_filepaths) > self.max_num_tracks:
                self.mix_dirs.remove(mix_dir)

        print(len(self.mix_dirs))
        self.mix_dirs = self.mix_dirs[indices[0] : indices[1]]  # select subset
        print(len(self.mix_dirs))
        self.items_since_load = self.buffer_reload_rate

    def reload_buffer(self):

        self.examples = []  # clear buffer
        self.items_since_load = 0  # reset iteration counter
        nbytes_loaded = 0  # counter for data in RAM

        # different subset in each
        random.shuffle(self.mix_dirs)

        # load files into RAM
        while nbytes_loaded < self.buffer_size_gb * 1e9:
            for mix_dir in tqdm(self.mix_dirs):
                mix_id = os.path.basename(mix_dir)
                mix_filepath = glob.glob(os.path.join(mix_dir, "*.wav"))[0]

                if "AimeeNorwich_Child" in mix_filepath:
                    continue

                # save only a random subset of this song so we can load more songs
                silent = True
                counter = 0
                while silent:

                    num_frames = torchaudio.info(mix_filepath).num_frames

                    offset = np.random.randint(
                        0,
                        num_frames - self.buffer_audio_length - 1,
                    )

                    # now check the length of the mix
                    y, sr = torchaudio.load(
                        mix_filepath,
                        frame_offset=offset,
                        num_frames=self.buffer_audio_length,
                    )

                    energy = (y**2).mean()
                    print(energy)
                    if energy > 1e-3:
                        silent = False

                    counter += 1
                    if counter > 10:
                        break

                if silent:
                    continue

                mix_num_frames = y.shape[-1]
                nbytes = y.element_size() * y.nelement()
                nbytes_loaded += nbytes

                # now find all the track filepaths
                track_filepaths = glob.glob(
                    os.path.join(mix_dir, f"{mix_id}_RAW", "*.wav")
                )

                # check length of each track
                tracks = []
                for tidx, track_filepath in enumerate(track_filepaths):
                    x, sr = torchaudio.load(
                        track_filepath,
                        frame_offset=offset,
                        num_frames=self.buffer_audio_length,
                    )
                    tracks.append(x)

                    nbytes = x.element_size() * x.nelement()
                    nbytes_loaded += nbytes

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

        print(len(self.examples))
        print()

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _):

        # increment counter
        self.items_since_load += 1

        # load next chunk into buffer if needed
        if self.items_since_load > self.buffer_reload_rate:
            self.reload_buffer()

        silent = True
        no_active_tracks = True
        while silent or no_active_tracks:
            # select an example at random
            example_idx = np.random.randint(0, len(self.examples))
            example = self.examples[example_idx]

            # get the whole mix
            y = example["mix_audio"].clone()
            energy = (y**2).mean()

            if y.shape[-1] == self.length and energy > 1e-3:
                silent = False
            else:
                continue

            # -------------------- load the tracks from RAM --------------------
            x = torch.zeros((self.max_num_tracks, self.length))
            pad = [True] * self.max_num_tracks  # note which tracks are empty
            random.shuffle(example["track_audio"])  # load random tracks each time

            track_energy = []
            for track_idx, track in enumerate(example["track_audio"]):
                energy = (track**2).mean()
                track_energy.append((track_idx, energy))

            # sort tracks based on energy
            track_energy = sorted(track_energy, key=lambda x: x[1], reverse=True)

            tidx = 0
            for track_idx, energy in track_energy:
                if energy > 1e-4:  # ensure track is active

                    x_s = example["track_audio"][track_idx].clone()

                    # could also try loudness normalization next
                    x_s /= x_s.abs().max().clamp(1e-6)
                    x_s *= 10 ** (-12 / 20.0)

                    # gain_dB = np.random.rand() * 12
                    # gain_factor = 1 if np.random.rand() > 0.5 else -1
                    # gain_dB *= gain_factor
                    # gain_lin = 10 ** (gain_dB / 20.0)

                    # x_s *= gain_lin

                    x[tidx, :] = x_s
                    pad[tidx] = False
                    tidx += 1

                if (tidx + 1) >= self.max_num_tracks:
                    break

            if tidx > 0:
                no_active_tracks = False

        assert tidx > 0  # no tracks are loaded

        return x, y, torch.tensor(pad)
