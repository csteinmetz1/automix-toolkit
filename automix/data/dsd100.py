import torch
from typing import Tuple


class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        indices: Tuple[int, int],
        num_examples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.indices = indices
        self.num_examples_per_epoch = num_examples_per_epoch

        # first get all tracks
        # note that we use a different dataset split than default

    def __len__(self):
        return

    def __getitem__(self, idx):
        return x, y
