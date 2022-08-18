import torch

class MixingSecretsDataset(torch.data.utils.Dataset):
    def __init__(self) -> None:
        super().__init__()
