import numpy as np
import os
import librosa
import soundfile
import scipy
from data_tools.utils import scaled_in

class Model():
    def __init__(self, model, sample_rate, min_duration, n_fft, hop_length, chunk_length):
        self.model = model
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        
    def process_mono(self, audio):
        stft_noisy = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag_noisy, phase = librosa.magphase(stft_noisy)
        
        mag_noisy_db = librosa.amplitude_to_db(mag_noisy, ref=np.max)
        input = scaled_in(mag_noisy_db)
        
        n_bins, n_frames = input.shape
        
        # padding
        pad_width = self.chunk_length - (n_frames % self.chunk_length)
        if pad_width < self.chunk_length:
            input = np.pad(input, ((0, 0), (0, pad_width)), mode='constant')
            mag_noisy = np.pad(mag_noisy, ((0, 0), (0, pad_width)), mode='constant')
            phase = np.pad(phase, ((0, 0), (0, pad_width)), mode='constant')
            
        n_padded = input.shape[1]
        num_chunks = n_padded // self.chunk_length
        
        mask = np.zeros(input.shape)
        
        for i in range(num_chunks):
            start = i * self.chunk_length
            end = start + self.chunk_length
            
            chunk = input[:, start: end]
            chunk = chunk.reshape(1, n_bins, self.chunk_length, 1)
            
            pred_mask = self.model.predict(chunk, verbose=0)
            mask[:, start: end] = pred_mask[0, :, :, 0]
            
        mag_clean = mag_noisy * mask
        stft_clean = mag_clean * phase
        
        audio_clean = librosa.istft(stft_clean, hop_length=self.hop_length)
        
        # trim padding
        if len(audio_clean) > len(audio):
            audio_clean = audio_clean[: len(audio)]
        elif len(audio_clean) < len(audio):
            audio = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
            
        return audio_clean
    
    def predict(self, artifacts_path, audio_file, output_file, headroom=1.0):
        full_path = os.path.join(artifacts_path, audio_file)
        
        audio, sr = librosa.load(full_path, sr=self.sample_rate, mono=False)
        
        clean_audio = None
        
        if audio.ndim == 1:
            # mono
            print("Mono")
            clean_audio = self.process_mono(audio)
        else:
            # stereo
            print("Stereo")
            left_channel = audio[0]
            right_channel = audio[1]
            
            clean_left = self.process_mono(left_channel)
            clean_right = self.process_mono(right_channel)
            
            clean_audio = np.vstack([clean_left, clean_right]).T

        if headroom != 1.0:
            clean_audio = clean_audio * headroom
            
        output_path = os.path.join(artifacts_path, output_file)
        soundfile.write(output_path, clean_audio, self.sample_rate)
        print(f"Denoised and saved as {output_file}")
        
            
        