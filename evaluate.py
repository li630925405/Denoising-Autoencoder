import numpy as np
import os
import musdb
import librosa 
import soundfile as sf
import museval

pred_path = "./DL/Wave-U-Net-Pytorch/audio_examples/A_classic_Education/"

mus = musdb.DB(root="./database/musdb18", subsets="train")
# (nsrc, nsampl, nchan)
target_sources = np.zeros((4, 1000000))
origin_sources = np.zeros((4, 1000000))
noise_sources = np.zeros((4, 1000000))
output_sources = np.zeros((4, 1000000))

for track in mus:
    # ['mixture', 'drums', 'bass', 'other', 'vocals']
    tags = ['drums', 'bass', 'other', 'vocals']
    origin_path = pred_path + track.name + '.wav'
    noise_path = pred_path + track.name + "_noise" + '.wav'
    output_path = pred_path + track.name + "_output" + '.wav'
    for i, tag in enumerate(tags):
        # out_name = pred_path + track.name + '.wav'
        # print(out_name)
        # sf.write(out_name, track.audio[1000000:2000000], track.rate)
        target_sources[i] = track.targets[tag].audio[1000000:2000000, 0]
        origin_sources[i], sr = librosa.load(origin_path + '_' + tag + '.wav', sr=44100)
        tmp, sr = librosa.load(noise_path + '_' + tag + '.wav', sr=44100)
        noise_sources[i] = tmp[0:1000000]
        tmp, sr = librosa.load(output_path + '_' + tag + '.wav', sr=44100)
        output_sources[i] = np.concatenate((tmp, np.zeros((1000000 - tmp.shape[0]))))
    break

res = {'origin':origin_sources, 'noise':noise_sources, 'output':output_sources}

for type in ['origin', 'noise', 'output']:
    print("type: ", type)
    sources = np.expand_dims(res[type], axis=2)
    SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, sources)
    print("SDR: ", np.mean(SDR), "ISR: ", np.mean(ISR), "SIR: ", np.mean(SIR), "SAR: ", np.mean(SAR))