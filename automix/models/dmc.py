import torch

class DifferentiableMixingConsole(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x