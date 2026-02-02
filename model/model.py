from args import parser
import numpy as np
import os
import librosa
import soundfile
from prepare_data.utils import create_data
from model_training.utils import training
from prediction.utils import prediction
import scipy
from data_tools.utils import scaled_in, inv_scaled_ou
from data_tools.utils import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio

class Model():
    def __init__(self, model, weights, model_name, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):
        self.model = model
        self.weights = weights
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.frame_length = frame_length
        self.hop_length_frame = hop_length_frame
        self.n_fft = n_fft
        self.hop_length_fft = hop_length_fft
        
        if weights is not None and os.path.exists(weights):
            print(f"Loading weights from {weights}...")
            self.model.load_weights(weights)

    def predict(self, artifacts_path, audio_file, output_file, noise_coef=1.0, headroom=0.02):
        """
        1. Convert to Spectrogram (Linear & dB)
        2. Scale Input for Model
        3. Predict Mask (Sigmoid)
        4. Apply Mask: Linear_Mag * Mask
        5. Reconstruct with original Phase
        """
        # 1. Load Audio
        y, sr = librosa.load(f"{artifacts_path}/{audio_file}", sr=self.sample_rate)
         
        stft_noisy = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length_fft)
        mag_noisy, phase_noisy = librosa.magphase(stft_noisy)
        
        mag_noisy_db = librosa.amplitude_to_db(mag_noisy, ref=np.max)
        X_input = scaled_in(mag_noisy_db)
        
        n_bins, n_frames = X_input.shape
        chunk_width = 512 # The time dimension expected by U-Net
         
        # Pad spectrogram to be divisible by chunk_width
        pad_width = chunk_width - (n_frames % chunk_width)
        if pad_width < chunk_width:
             X_input = np.pad(X_input, ((0, 0), (0, pad_width)), mode='constant')
             # Also pad magnitude and phase for reconstruction later
             mag_noisy = np.pad(mag_noisy, ((0, 0), (0, pad_width)), mode='constant')
             phase_noisy = np.pad(phase_noisy, ((0, 0), (0, pad_width)), mode='constant')
        
        n_frames_padded = X_input.shape[1]
        num_chunks = n_frames_padded // chunk_width
        
        # Placeholder for the reconstructed mask
        full_mask = np.zeros(X_input.shape)
        
        print(f"Processing {num_chunks} chunks for masking strategy...")

        for i in range(num_chunks):
            start = i * chunk_width
            end = start + chunk_width
            
            # Extract chunk: Shape (512, 512)
            chunk = X_input[:, start:end]
            
            # Reshape for model: (1, 512, 512, 1)
            chunk = chunk.reshape(1, n_bins, chunk_width, 1)
            
            # Predict Mask (Sigmoid output [0, 1])
            pred_mask = self.model.predict(chunk, verbose=0)
            
            # Remove batch dim: (512, 512)
            pred_mask = pred_mask[0, :, :, 0]
            
            # Store into full mask array
            full_mask[:, start:end] = pred_mask
            
        mag_clean = mag_noisy * full_mask
        stft_clean = mag_clean * phase_noisy
        
        audio_clean = librosa.istft(stft_clean, hop_length=self.hop_length_fft)
        
        if len(audio_clean) > len(y):
             audio_clean = audio_clean[:len(y)]
        elif len(audio_clean) < len(y):
             # This rarely happens with padding, but good to handle
             audio_clean = np.pad(audio_clean, (0, len(y) - len(audio_clean)))            
            
        final_audio = audio_clean
        # Apply headroom if specified
        if headroom != 1.0:
            final_audio = final_audio * headroom
        
        
        print(f"Output audio shape: {final_audio.shape}")
        
        soundfile.write(artifacts_path + "/" + output_file, final_audio, self.sample_rate)
        return final_audio
    
    def combine_signals(self, clean_pred, original_noisy, sr=44100, crossover=6000, high_freq_vol=0.6):
        sos = scipy.signal.butter(10, crossover, 'hp', fs=sr, output='sos')
        high_freq_noise = scipy.signal.sosfilt(sos, original_noisy)
        
        sos_lp = scipy.signal.butter(10, crossover, 'lp', fs=sr, output='sos')
        low_freq_clean = scipy.signal.sosfilt(sos_lp, clean_pred)
        
        return low_freq_clean + (high_freq_noise * high_freq_vol)
    