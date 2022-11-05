import torch

from automix.models.dmc import Mixer

mixer = Mixer(44100)

bs = 1
num_tracks = 2
samp = 5

x = torch.tensor(
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.9, 0.9, 0.9, 0.9],
        [0.2, 0.2, 0.2, 0.2],
    ]
)
x = x.view(2, 2, -1)
p = torch.tensor(
    [[[0.5, 1.0], [0.5, 0.0]], [[0.5, 0.0], [0.5, 1.0]]],
)
p = p.view(2, 2, -1)

print(x)

y = mixer(x, p)

print(y.shape)
print(y)
