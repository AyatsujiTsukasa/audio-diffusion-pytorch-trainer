import os
import numpy as np
from torch.utils.data import Dataset
from librosa.core import load

class AudioPreprocessingDataset(Dataset):
    SEG_LENGTH = 2048 / 3000 * 16
    def __init__(self, audio2mel_df, sr):
        self.audio2mel_df = audio2mel_df
        self.sr = sr
        self.audio_cache = dict()

    def __len__(self):
        return len(self.audio2mel_df)

    def __load_audio_file(self, file_path):
        if file_path in self.audio_cache:
            return self.audio_cache[file_path]

        self.audio_cache.clear()
        data, _ = load(file_path, sr=self.sr, mono=False)

        self.audio_cache[file_path] = data
        return data

    def __load_wav_to_torch(self, file_path, channel, start_sec, end_sec):
        data = self.__load_audio_file(file_path)

        seg_sample_len = int(AudioPreprocessingDataset.SEG_LENGTH * self.sr)
        data = data[channel, int(start_sec * self.sr):int(end_sec * self.sr)]
        diff = seg_sample_len - len(data)
        if diff > 0:
            data = np.pad(data, (0, diff), mode='constant', constant_values=0)
        elif diff < 0:
            data = data[:diff]
        
        return data

    def __getitem__(self, idx):
        row = self.audio2mel_df.iloc[idx]
        file_path = row['path']
        channel = row['channel']
        start_sec = row['start']
        end_sec = row['end']
        label = str(row['label'])

        aud = self.__load_wav_to_torch(file_path, channel, start_sec, end_sec)
        wav_name = f"{os.path.splitext(os.path.basename(file_path))[0]}-CH{channel}-{start_sec:.2f}~{end_sec:.2f}-{label}.wav"
        
        return wav_name, label, aud
