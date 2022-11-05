import torch


class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.root_dir = root_dir
        

    def __len__(self):
        return

    def __getitem__(self, idx):
        return x, y
