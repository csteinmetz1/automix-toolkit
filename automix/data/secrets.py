import torch


class MixingSecretsDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
