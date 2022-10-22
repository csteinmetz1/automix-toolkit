import torch

from automix.models.dmc import DifferentiableMixingConsole


sr = 44100
seq_len = 131072
bs = 3
num_tracks = 8


x = torch.randn(bs, num_tracks, seq_len)
print(x.shape)

model = DifferentiableMixingConsole(sr)

y, p = model(x)

print(y.shape, p.shape)
