import librosa
import numpy as np
import musdb
import soundfile as sf

noise_path = "./database/noise/"
block, sr = librosa.load("./database/Block.wav", sr=44100, mono=False)
block = block.transpose()
chatter, sr = librosa.load("./database/chatter.mp3", sr=44100, mono=False)
chatter = chatter.transpose()

mus = musdb.DB(root="./database/musdb18", subsets="train")

for track in mus:
    src = track.audio[1000000:2000000, :]
    left = np.convolve(src[:, 0], block[:, 0])
    right = np.convolve(src[:, 1], block[:, 1])
    src_block = np.vstack([left, right]).transpose()
    print("src shape: ", src.shape)
    print("left shape: ", left.shape)
    print("block shape: ", block.shape)
    # res = chatter[:, 0:src_block.shape[1]] / 10 + src_block / 10
    res = chatter[0:src_block.shape[0], :] / 10 + src_block / 5
    res = res[0:1000000, :]
    sf.write(noise_path + track.name + ".wav", res, track.rate)
    # sf.write(track.name + ".wav", track.audio, track.rate)
    print(track.path)