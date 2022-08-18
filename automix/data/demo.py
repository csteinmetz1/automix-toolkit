import torch


class DemoMultitrackDataset(torch.data.utils.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return

    def __getitem__(self, idx):
        return x, y

