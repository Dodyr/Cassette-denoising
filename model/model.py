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
        
    # def combine_signals(self, clean_pred, original_noisy, sr=44100, crossover=6000, high_freq_vol=0.6):
    #     sos = scipy.signal.butter(10, crossover, 'hp', fs=sr, output='sos')
    #     high_freq_noise = scipy.signal.sosfilt(sos, original_noisy)
        
    #     sos_lp = scipy.signal.butter(10, crossover, 'lp', fs=sr, output='sos')
    #     low_freq_clean = scipy.signal.sosfilt(sos_lp, clean_pred)
        
    #     result = low_freq_clean + (high_freq_noise * high_freq_vol) 
        
    #     return result

    def predict(self, artifacts_path, audio_file, output_file, noise_coef=1.0, headroom=0.02):
        """
        Predict and denoise audio with robust reconstruction and normalization.
        
        Args:
            artifacts_path: Path to audio files
            audio_file: List of audio filenames to process
            output_file: Output filename
            a_in, b_in: Input scaling parameters (must match training, default 40.0)
            a_ou, b_ou: Output scaling parameters (must match training, default 40.0)
            noise_coef: Coefficient to control denoising strength (default 0.85, range 0-1)
            headroom: Amplitude factor to apply (1.0 = no change, 0.8 = 80% of peak, etc., by default is 0.02, because audio in the output is 45 timnes louder)
        """
        audio = audio_files_to_numpy(artifacts_path, audio_file, self.sample_rate,
                                      self.frame_length, self.hop_length_frame, self.min_duration)
         
        dim_square_spec = int(self.n_fft / 2) + 1
        print(f"Spectrogram dimension: {dim_square_spec}")
        
        m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
            audio, dim_square_spec, self.n_fft, self.hop_length_fft)
        
        # Global scaling to have distribution -1/1 (MUST match training parameters!)
        X_in = scaled_in(m_amp_db_audio)
        # Reshape for prediction
        X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
        # Prediction using loaded network
        X_pred = self.model.predict(X_in)
        # Rescale back the noise model (MUST match training parameters!)
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        # Remove noise model from noisy speech with noise_coef control
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0] * noise_coef
        # X_denoise = inv_sca_X_pred[:, :, :, 0]
        # For audio not to explode
        # X_denoise = librosa.db_to_amplitude(X_denoise, ref=1.0)
         
        print(f"Denoised spectrogram shape: {X_denoise.shape}")
        print(f"Phase shape: {m_pha_audio.shape}")
        
        # Reconstruct audio frames from denoised spectrograms
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(
            X_denoise, m_pha_audio, self.frame_length, self.hop_length_fft)
        
        nb_samples = audio_denoise_recons.shape[0]

        if self.hop_length_frame == self.frame_length:
            print("No overlap detected (hop_length_frame == frame_length). Using simple concatenation.")
            denoise_long = audio_denoise_recons.flatten()
            # original_long = audio.flatten()
        else:
            print(f"Overlap detected. Using Hann window Overlap-Add with hop_length_frame={self.hop_length_frame}")
            
            # Create Hann window for overlap-add
            window = np.hanning(self.frame_length)
            
            # Calculate output length
            output_length = (nb_samples - 1) * self.hop_length_frame + self.frame_length
            denoise_long = np.zeros(output_length)
            # original_long = np.zeros(output_length)
            window_sum = np.zeros(output_length)
            
            # Overlap-add with windowing
            for i in range(nb_samples):
                start = i * self.hop_length_frame
                end = start + self.frame_length
                
                windowed_frame = audio_denoise_recons[i, :] * window
                # original_frame = audio[i, :] * window
                
                if end <= output_length:
                    denoise_long[start:end] += windowed_frame
                    # original_long[start:end] += original_frame
                    window_sum[start:end] += window
            
            # Normalize by window sum (compensate for overlapping windows)
            # Avoid division by zero
            denoise_long = np.divide(denoise_long, window_sum, where=window_sum > 1e-6, out=denoise_long)
            # original_long = np.divide(original_long, window_sum, where=window_sum > 1e-6, out=original_long)
        
        # print(f"Applying High-Frequency mixing: >{crossover_freq}Hz at vol {high_freq_vol}")
        # final_audio = self.combine_signals(denoise_long, original_long, 
        #                                    self.sample_rate, crossover_freq, high_freq_vol)
        final_audio = denoise_long
        # Apply headroom if specified
        if headroom != 1.0:
            final_audio = final_audio * headroom
        
        
        print(f"Output audio shape: {final_audio.shape}")
        print(f"Output amplitude range: [{np.min(final_audio):.4f}, {np.max(denoise_long):.4f}]")
        
        soundfile.write(artifacts_path + "/" + output_file, final_audio, self.sample_rate)
    