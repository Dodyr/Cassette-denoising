import numpy as np
import os
import librosa
import soundfile
import scipy
from data_tools.utils import scaled_in
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

    def _process_single_channel(self, audio_1d):
        """
        Helper function to process a single channel (Mono logic)
        """
        # 1. STFT
        stft_noisy = librosa.stft(audio_1d, n_fft=self.n_fft, hop_length=self.hop_length_fft)
        mag_noisy, phase_noisy = librosa.magphase(stft_noisy)
        
        # 2. Prepare Input (dB -> scale)
        mag_noisy_db = librosa.amplitude_to_db(mag_noisy, ref=np.max)
        X_input = scaled_in(mag_noisy_db)
        
        n_bins, n_frames = X_input.shape
        chunk_width = 512 
         
        # 3. Padding logic
        pad_width = chunk_width - (n_frames % chunk_width)
        if pad_width < chunk_width:
             X_input = np.pad(X_input, ((0, 0), (0, pad_width)), mode='constant')
             mag_noisy = np.pad(mag_noisy, ((0, 0), (0, pad_width)), mode='constant')
             phase_noisy = np.pad(phase_noisy, ((0, 0), (0, pad_width)), mode='constant')
        
        n_frames_padded = X_input.shape[1]
        num_chunks = n_frames_padded // chunk_width
        
        full_mask = np.zeros(X_input.shape)
        
        # 4. Prediction Loop
        for i in range(num_chunks):
            start = i * chunk_width
            end = start + chunk_width
            
            chunk = X_input[:, start:end]
            chunk = chunk.reshape(1, n_bins, chunk_width, 1)
            
            # Predict
            pred_mask = self.model.predict(chunk, verbose=0)
            full_mask[:, start:end] = pred_mask[0, :, :, 0]
            
        # 5. Apply Mask & Reconstruct
        mag_clean = mag_noisy * full_mask
        stft_clean = mag_clean * phase_noisy
        
        audio_clean = librosa.istft(stft_clean, hop_length=self.hop_length_fft)
        
        # Trim padding
        original_len = len(audio_1d)
        if len(audio_clean) > original_len:
             audio_clean = audio_clean[:original_len]
        elif len(audio_clean) < original_len:
             audio_clean = np.pad(audio_clean, (0, original_len - len(audio_clean)))            
            
        return audio_clean

    def predict(self, artifacts_path, audio_file, output_file, noise_coef=1.0, headroom=0.02):
        """
        Main prediction method handling both Mono and Stereo files.
        """
        full_path = os.path.join(artifacts_path, audio_file)
        print(f"Processing: {full_path}")
        
        # 1. Load Audio with mono=False to preserve channels
        y, sr = librosa.load(full_path, sr=self.sample_rate, mono=False)
        
        final_audio = None
        
        # 2. Check dimensions
        if y.ndim == 1:
            # --- MONO CASE ---
            print("Mode: Mono")
            final_audio = self._process_single_channel(y)
        else:
            # --- STEREO CASE ---
            print("Mode: Stereo (Processing Left and Right channels separately)")
            # y shape is (2, samples)
            left_channel = y[0]
            right_channel = y[1]
            
            print("-> Processing Left Channel...")
            clean_left = self._process_single_channel(left_channel)
            
            print("-> Processing Right Channel...")
            clean_right = self._process_single_channel(right_channel)
            
            # Stack back to (samples, 2) for soundfile
            final_audio = np.vstack([clean_left, clean_right]).T
        
        # 3. Apply Headroom (Post-processing)
        if headroom != 1.0:
            final_audio = final_audio * headroom
        
        print(f"Output audio shape: {final_audio.shape}")
        
        output_path = os.path.join(artifacts_path, output_file)
        soundfile.write(output_path, final_audio, self.sample_rate)
        print(f"Saved to: {output_path}")
        
        return final_audio
    
    def combine_signals(self, clean_pred, original_noisy, sr=44100, crossover=6000, high_freq_vol=0.6):
        sos = scipy.signal.butter(10, crossover, 'hp', fs=sr, output='sos')
        high_freq_noise = scipy.signal.sosfilt(sos, original_noisy)
        
        sos_lp = scipy.signal.butter(10, crossover, 'lp', fs=sr, output='sos')
        low_freq_clean = scipy.signal.sosfilt(sos_lp, clean_pred)
        
        return low_freq_clean + (high_freq_noise * high_freq_vol)