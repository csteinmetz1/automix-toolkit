import os
import glob
import torch
import torchaudio
import numpy as np

from tqdm import tqdm
from typing import Tuple, List


class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        length: int,
        sample_rate: float,
        indices: Tuple[int, int],
        track_names: List[str] = ["bass", "drums", "other", "vocals"],
        num_examples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.length = length
        self.sample_rate = sample_rate
        self.indices = indices
        self.track_names = track_names
        self.num_examples_per_epoch = num_examples_per_epoch

        # first get all tracks
        mix_dirs = glob.glob(os.path.join(root_dir, "Mixtures", "**", "*"))
        mix_dirs = [md for md in mix_dirs if os.path.isdir(md)]
        mix_dirs = sorted(mix_dirs)
        # note that we use a different dataset split than default
        self.mix_dirs = mix_dirs[indices[0] : indices[1]]

        self.examples = []
        # iterate over mix directories to check lengths
        for mix_dir in tqdm(self.mix_dirs):
            # find mix
            mix_id = os.path.basename(mix_dir)
            subset = os.path.basename(os.path.dirname(mix_dir))
            mix_filepath = os.path.join(mix_dir, "mixture.wav")

            num_frames = torchaudio.info(mix_filepath).num_frames

            # now find tracks
            track_filepaths = []
            for track_name in track_names:
                track_filepath = os.path.join(
                    root_dir, "Sources", subset, mix_id, f"{track_name}.wav"
                )
                track_filepaths.append(track_filepath)

            self.examples.append(
                {
                    "mix_id": mix_id,
                    "mix_filepath": mix_filepath,
                    "num_frames": num_frames,
                    "track_filepaths": track_filepaths,
                }
            )

        print(
            f"Found {len(mix_dirs)} mixes. Using {len(self.mix_dirs)} in this subset."
        )

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, idx):
        # select a mix at random
        mix_idx = np.random.randint(0, len(self.mix_dirs))
        example = self.examples[mix_idx]

        # load the chunk of the mix
        silent = True
        while silent:
            # get random offset
            offset = np.random.randint(0, example["num_frames"] - self.length - 1)

            y, sr = torchaudio.load(
                example["mix_filepath"],
                frame_offset=offset,
                num_frames=self.length,
            )
            energy = (y**2).mean()
            if energy > 1e-6:
                silent = False

        y /= y.abs().max().clamp(1e-8)  # peak normalize

        # -------------------- load the tracks from disk --------------------
        x = torch.zeros((len(self.track_names) * 2, self.length))
        pad = [True] * len(self.track_names) * 2  # note which tracks are empty

        tidx = 0
        for track_filepath in example["track_filepaths"]:
            x_s, sr = torchaudio.load(
                track_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )

            # split into left and right (since we have stereo)
            x_s_L = x_s[0:1, :]
            x_s_R = x_s[1:2, :]

            for x_s_ch in [x_s_L, x_s_R]:
                x_s_ch /= x_s_ch.abs().max().clamp(1e-6)
                gain_dB = np.random.rand() * 18  # apply random gain
                x_s_ch *= 10 ** (-gain_dB / 20.0)
                x[tidx, :] = x_s_ch
                pad[tidx] = False
                tidx += 1

        return x, y, pad
