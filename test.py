from network import UNet
import torch
import torchaudio
import torch.nn as nn
import os
import time

#build dataset
dataset_path = "D:/songs_headphones/string_qt_val_cut"
n_files = 0
fs = 0
pad = nn.ConstantPad1d((0,1),0)
start_time = time.time()
for subdir, dirs, files in os.walk(dataset_path):

    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".wav"):
            if n_files == 0:
                dataset,fs = torchaudio.load(filepath)
                dataset = pad(dataset)

                #dataset = dataset/torch.max(dataset)
            elif n_files == 1:
                new_file,fs = torchaudio.load(filepath)
                new_file = pad(new_file)
                #new_file = new_file/torch.max(new_file)

                dataset = torch.stack((dataset,new_file))
            else:
                new_file,fs = torchaudio.load(filepath)
                new_file = pad(new_file)
                #new_file = new_file/torch.max(new_file)

                dataset = torch.cat((dataset,new_file.unsqueeze(0)))

            n_files = n_files + 1

print("finished loading: {} files loaded, Total Time: {}".format(n_files, time.time()-start_time))

G = UNet(1,2)
G.cuda()

#load the model
G.load_state_dict(torch.load("g_param.pth"))

G.eval()

results_path = "val_out"

for j in range(dataset.size()[0]):
    input_stereo = dataset[j,:,:].cuda()
    input_wav = torch.mean(input_stereo, dim=0).unsqueeze(0)
    output_wav = G(input_wav.unsqueeze(0)).cpu().detach()
    torchaudio.save(results_path + os.sep + "test_output_" + str(j) + ".wav", output_wav.squeeze(),fs)
