import os
from tqdm import tqdm
import soundfile as sf
from torch.utils.data import DataLoader

import argparse
import pandas as pd
from pathlib import Path

from dataset.audio_dataset import AudioPreprocessingDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    return args


def preprocess_wavs(annotation_list, batch_size, sr, wav_out_folder):
    dataset = AudioPreprocessingDataset(annotation_list, sr=sr)
    print(f"Preprocessing {len(dataset)} samples...")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    n = 0
    for x_t in tqdm(data_loader):
        wav_names, labels, aud = x_t
        for label in labels:
            mel_label_path = wav_out_folder / label
            mel_label_path.mkdir(exist_ok=True, parents=True)

        for i, wav in enumerate(aud):
            label = labels[i]
            wav_out_path = wav_out_folder / label / wav_names[i]
            if os.path.exists(wav_out_path):
                print(f"File {wav_out_path} already exists")
            else:
                sf.write(str(wav_out_path), wav, sr)

            n += 1

    print(f"Created {n} samples...")


def main():
    args = parse_args()
    model_root_path = args.models_root_path / 'ADPT' / args.model_name
    annotation_list_path = model_root_path / 'annotation_list.csv'

    annotation_list = pd.read_csv(annotation_list_path)
    wav_out_folder = model_root_path / 'preprocessed_wavs'
    wav_out_folder.mkdir(exist_ok=True, parents=True)

    preprocess_wavs(annotation_list, args.batch_size, 48000, wav_out_folder)


if __name__ == '__main__':
    main()
