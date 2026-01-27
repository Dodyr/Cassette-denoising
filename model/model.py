from args import parser
import numpy as np
import os
import librosa
import soundfile
from prepare_data.utils import create_data
from model_training.utils import training
from prediction.utils import prediction
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

    def predict(self, artifacts_path, audio_file, output_file):
        audio = audio_files_to_numpy(artifacts_path, audio_file, self.sample_rate,
                                      self.frame_length, self.hop_length_frame, self.min_duration)
         
        dim_square_spec = int(self.n_fft / 2) + 1
        print(dim_square_spec)
        
        m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, self.n_fft, self.hop_length_fft)
        
        #global scaling to have distribution -1/1
        X_in = scaled_in(m_amp_db_audio)
        #Reshape for prediction
        X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
        #Prediction using loaded network
        X_pred = self.model.predict(X_in)
        #Rescale back the noise model
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        #Remove noise model from noisy speech
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
        print(X_denoise.shape)
        print(m_pha_audio.shape)
        print(self.frame_length)
        print(self.hop_length_fft)
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, self.frame_length, self.hop_length_fft)
        #Flatten frames into continuous audio with overlap-add
        def overlap_add(frames, hop_length):
            n_frames, frame_len = frames.shape
            output_len = frame_len + (n_frames - 1) * hop_length
            output = np.zeros(output_len, dtype=frames.dtype)
            for i in range(n_frames):
                start = i * hop_length
                output[start:start+frame_len] += frames[i]
            return output
        denoise_long = overlap_add(audio_denoise_recons, hop_length=self.hop_length_frame)
        denoise_long /= np.max(np.abs(denoise_long)) + 1e-7
        # nb_samples = audio_denoise_recons.shape[0]
        # #Save all frames in one file
        # denoise_long = audio_denoise_recons.reshape(1, nb_samples * self.frame_length)*10
        soundfile.write(artifacts_path + "/" + output_file, denoise_long, self.sample_rate)
    