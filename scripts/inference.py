import os
import glob
import torch
import argparse
import torchaudio

from automix.system import System

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_path",
        help="Path to pretrained model checkpoint.",
        type=str,
    )
    parser.add_argument(
        "track_dir",
        help="Path to directory containing  model checkpoint.",
        type=str,
    )
    parser.add_argument(
        "--ext",
        help="Expected file extension for audio files.",
        default="wav",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="Path and filename to store output mix.",
        default="./mix.wav",
        type=str,
    )
    parser.add_argument(
        "--max_samples",
        help="Maximum number of samples to process.",
        default=262144,
        type=str,
    )

    args = parser.parse_args()

    # load pretrained model
    system = System.load_from_checkpoint(args.ckpt_path)
    system.eval()

    # load the input tracks
    track_filepaths = glob.glob(os.path.join(args.track_dir, f"*.{args.ext}"))
    track_filepaths = sorted(track_filepaths)
    tracks = []
    for track_filepath in track_filepaths:
        x, sr = torchaudio.load(track_filepath)
        x = x[:, : args.max_samples]
        tracks.append(x)

    # add dummy tracks of silence if needed

    tracks = torch.stack(tracks, dim=0)
    tracks = tracks.permute(1, 0, 2)
    print(tracks.shape)
    # tracks have shape (1, num_tracks, seq_len)

    # pass to the model
    mix, params = system(tracks)

    for param in zip(track_filepaths, params.squeeze()):
        print(track_filepath, param)

    # save out the mix
    mix_filepath = args.output
    mix /= mix.abs().max()
    print(mix.shape)
    torchaudio.save(mix_filepath, mix.view(2, -1), sr)
