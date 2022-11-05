import torch
from tqdm import tqdm
from automix.data.secrets import MixingSecretsDataset

if __name__ == "__main__":
    dataset = MixingSecretsDataset(
        "/import/c4dm-datasets/mixing-secrets", 131072, 44100
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=16,
    )

    for idx, batch in enumerate(tqdm(dataloader), 0):
        x, y = batch
