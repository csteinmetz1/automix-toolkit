import os
import torch
import shutil
import torchaudio
import pytorch_lightning as pl
from tqdm import tqdm
from automix.data.medleydb import MedleyDBDataset
from automix.data.secrets import MixingSecretsDataset

pl.seed_everything(42)

root_dir = "/import/c4dm-datasets-ext/automix-toolkit/debug"
os.makedirs(root_dir, exist_ok=True)

if __name__ == "__main__":
    dataset = MedleyDBDataset(
        [
            "/import/c4dm-datasets/MedleyDB_V1/V1",
            "/import/c4dm-datasets/MedleyDB_V2/V2",
        ],
        262144,
        44100,
        max_num_tracks=16,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=16,
    )

    for idx, batch in enumerate(tqdm(dataset), 0):
        x, y, pad = batch
        print(idx, x.shape, y.shape, pad)

        batch_dir = os.path.join(root_dir, f"{idx:03d}")
        if os.path.isdir(batch_dir):
            shutil.rmtree(batch_dir)
        os.makedirs(batch_dir)

        mix_filepath = os.path.join(batch_dir, "mix.wav")
        torchaudio.save(mix_filepath, y.view(2, -1), 44100)

        for tidx, (x_s, mask) in enumerate(zip(x, pad)):
            if not mask:
                track_filepath = os.path.join(batch_dir, f"{tidx}.wav")
                torchaudio.save(track_filepath, x_s.view(1, -1), 44100)

        if idx > 5:
            break
