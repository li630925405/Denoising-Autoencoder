import torch
import librosa
import numpy as np
import soundfile

from module import Autoencoder

# Load state dict from the disk (make sure it is the same name as above)
state_dict = torch.load("./model/model_4c1p1k_interp.tar")

# Create a new model and load the state

model = Autoencoder()
model.load_state_dict(state_dict)

filename = "/homes/yl021/database/noise/A Classic Education - NightOwl.wav"
# filename = "/homes/yl021/database/noise/Actions - Devil's Words.wav"
output_path = "/homes/yl021/database/output/A Classic Education - NightOwl_4c1p1k_interp.wav"

y, sr = librosa.load(filename, 44100)
spec = librosa.stft(y[0:1000000], n_fft=1024, window='hann', win_length=1024,hop_length=512)
if spec.shape[1] % 2 == 0:
    spec = spec[:, 0:-1]
mags = np.abs(spec)
phase = spec / mags

pred = model(torch.Tensor(mags.reshape(1, 1, spec.shape[0], spec.shape[1])))
pred = np.squeeze(pred).detach().numpy()
if pred.shape[0] % 2 == 0:
    pred = pred[0:-1, 0:-1]

x = (spec.shape[0] - pred.shape[0] + 1) // 2
y = (spec.shape[1] - pred.shape[1] + 1) // 2
phase = phase[x:spec.shape[0]-x, y:spec.shape[1]-y]

# print(spec.shape)
# print(pred.shape)
# print(phase.shape)

print(mags)
print(pred)
print(sr)

with open('./spectrogram/origin.npy', 'wb') as f:
    np.save(f, mags)
with open('./spectrogram/pred.npy', 'wb') as f:
    np.save(f, pred)

audio_remade = librosa.istft(pred * phase, n_fft=1024, window='hann', win_length=1024,hop_length=512)

soundfile.write(output_path, audio_remade, sr)