import torch
from tqdm import tqdm

from automix.data.drums import ENSTDrumsDataset
from automix.models.mixwaveunet.mixwaveunet import MixWaveUNet

if __name__ == "__main__":
    length = 131072
    dataset = ENSTDrumsDataset("/import/c4dm-datasets/ENST-drums", length, 44100)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=16,
    )

    model = MixWaveUNet(7, length)

    for idx, batch in enumerate(tqdm(dataloader), 0):
        x, y = batch

        with torch.no_grad():
            y_hat = model(x)
