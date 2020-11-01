#converts full songs to smaller subsections, format: <song name>_<split number>.wav

import torch
import torchaudio
import math
import os

ext = ".wav"
directory = "acappella"
out_directory = "acappella-cut"

new_fs = 16000

#parameters for stft
n_fft = 511
hop_sz = int((n_fft+1)/2)
window_fn = torch.hann_window(n_fft)

for subdir, dirs, files in os.walk(directory):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".wav"):
            filename = file[:-4]

            wavData, fs = torchaudio.load(filepath)

            #print(wavData.size())
            if fs != 16000:
                print("started resampling " + file)
                wavData = torchaudio.transforms.Resample(fs, new_fs)(wavData)
                print("finished resampling " + file)
            else:
                print(file + " is already at appropriate fs")
            #print(wavData.size())

            #split into 256 frame chunks
            n_samps = hop_sz*256
            in_samps = wavData.size()[-1]

            n_out_files = math.floor(in_samps/n_samps)
            for i in range(n_out_files):
                out_audio = wavData[:,i*n_samps:((i+1)*n_samps)-1]
                torchaudio.save(out_directory + os.sep + filename + "_" + str(i) + ext, out_audio, new_fs)

            #complex_mix = torch.stft(out_audio, n_fft = n_fft, hop_length = hop_sz, window = window_fn)
            #complex_mix_pow = complex_mix.pow(2).sum(-1)
            #complex_mix_mag = torch.sqrt(complex_mix_pow)
            #print(complex_mix_mag.size())
