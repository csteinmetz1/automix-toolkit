import os
import glob
import torch
import argparse
import torchaudio
import numpy as np

from automix.evaluation.utils_evaluation import get_features

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_path",
        help="Path to the ground truth mix",
        type=str,
    )
    parser.add_argument(
        "automix_path",
        help="Path to generated mix",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="Path and filename to store features",
        default="./features.npy",
        type=str,
    )
    parser.add_argument(
        "--max_samples",
        help="Maximum number of samples to process.",
        default=1323000,
        type=int,
    )

    args = parser.parse_args()
    
    max_samples = args.max_samples
    mix_target_path = args.target_path
    automix_path = args.automix_path
    output_file = args.output
    
    print('\n Computing features for: \n ')
    print(mix_target_path)
    print(automix_path)
    
    # Load audio files
    
    target_audio, sr_target = torchaudio.load(mix_target_path)
    automix_audio, sr_automix = torchaudio.load(automix_path)
    
    # Trims audio files
    assert sr_target == sr_automix
    sr = sr_target
    start_sample = 0 * sr
    end_sample = start_sample + max_samples
    
    target_audio = target_audio[:, start_sample: end_sample]
    automix_audio = automix_audio[:, start_sample: end_sample]
    target_audio = target_audio.numpy()
    automix_audio = automix_audio.numpy()
    
    # Computes features
    features = get_features(target_audio, automix_audio)
    
    print('\n Features: \n')
    for k, i in features.items():
        print(k, i)  

    # save features
    
    np.save(output_file, features)
    print('\n Saving to : \n')
    print(output_file)