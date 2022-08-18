import torch

class MedleyDBDataset(torch.data.utils.Dataset):
    def __init__(self) -> None:
        super().__init__()
