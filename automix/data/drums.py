import os
import glob
import torch
import torchaudio
import numpy as np

from typing import List


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
        num_examples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.length = length
        self.sample_rate = sample_rate
        self.drummers = drummers
        self.track_names = track_names
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

        # load the chunk of the mix
        y, sr = torchaudio.load(
            mix_filepath,
            frame_offset=offset,
            num_frames=self.length,
        )

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
            x.append(x_s)
        x = torch.cat(x)

        return x, y
