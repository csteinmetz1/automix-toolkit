import torch
from tqdm import tqdm
from automix.data.dsd100 import DSD100Dataset

if __name__ == "__main__":
    dataset = DSD100Dataset("/import/c4dm-datasets-ext/DSD100", 262144, 44100, [0, 80])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )

    for idx, batch in enumerate(tqdm(dataloader), 0):
        x, y, pad = batch

        print(x.shape, y.shape)
