#TODO: Make the dataset bigger!, find the output dims of the network
#TODO: Replace loss with perceptual loss

from network import UNet
import torch
import torchaudio
import torch.nn as nn
import os
import time

n_fft = 1023
window_tensor = torch.hann_window(n_fft).cuda()

def mean(list):
    return sum(list)/len(list)

#TODO: Make this function work
def perceptual_loss(output,target):
    #print("start loss:")
    #build the spectrograms here (this isn't going to work right, i need to find the right parameters) WAS ONESIDED = TRUE
    complex_mix_output = torch.stft(output, n_fft = n_fft, hop_length = 128, window = window_tensor)
    complex_mix_pow_output = complex_mix_output.pow(2).sum(-1)
    complex_mix_mag_output = torch.sqrt(complex_mix_pow_output)
    #print("output max: {}, output min: {}".format(torch.max(complex_mix_mag_output),torch.min(complex_mix_mag_output)))

    complex_mix_target = torch.stft(target, n_fft = n_fft, hop_length = 128, window = window_tensor)
    complex_mix_pow_target = complex_mix_target.pow(2).sum(-1)
    complex_mix_mag_target = torch.sqrt(complex_mix_pow_target)
    #print("target max: {}, target min: {}".format(torch.max(complex_mix_mag_target),torch.min(complex_mix_mag_target)))

    comparison = nn.MSELoss()
    #print(comparison(complex_mix_mag_output,complex_mix_mag_target))
    return comparison(complex_mix_mag_output,complex_mix_mag_target)

#build dataset
dataset_path = "D:/songs_headphones/acappella-cut"
n_files = 0
fs = 0
pad = nn.ConstantPad1d((0,1),0)
for subdir, dirs, files in os.walk(dataset_path):

    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".wav"):
            if n_files == 0:
                dataset,fs = torchaudio.load(filepath)
                dataset = pad(dataset)

                dataset = dataset/torch.max(dataset)
            elif n_files == 1:
                new_file,fs = torchaudio.load(filepath)
                new_file = pad(new_file)
                new_file = new_file/torch.max(new_file)

                dataset = torch.stack((dataset,new_file))
            else:
                new_file,fs = torchaudio.load(filepath)
                new_file = pad(new_file)
                new_file = new_file/torch.max(new_file)

                dataset = torch.cat((dataset,new_file.unsqueeze(0)))

            n_files = n_files + 1

print("finished loading: {} files loaded".format(n_files))

#setup the network (refer to the github page)
#network input/output (N,C,H,W) = (1,1,256,256) => (1,2,256,256)
model = UNet(1,2)
model.cuda()
model.train()

criterion = torch.optim.Adam(model.parameters(), lr = .0001, betas = (.5,.999))

#for each epoch:
keep_training = True
training_losses = []
counter = 1

results_path = "output"

print("training start!")
while keep_training:
    epoch_losses = []
    start_time = time.time()
    #do TiFGAN transformations to data
    for j in range(dataset.size()[0]):
        #file,channel,waveform
        input_stereo = dataset[j,:,:].cuda()
        #torch.mean is torchaudio.downmix_mono
        input_wav = torch.mean(input_stereo, dim=0).unsqueeze(0)

        model.zero_grad()

        #run the data through the network
        output_wav = model(input_wav.unsqueeze(0))
        #do L1 (L2?, MSE?, crossEntropy?) loss on the network output
        loss = perceptual_loss(output_wav.squeeze(),input_stereo.squeeze())
        loss.backward()
        criterion.step()

        epoch_losses.append(loss.item())

    if counter%5 == 1:
        with torch.no_grad():
            for j in range(0,dataset.size()[0],100):
                #file,channel,waveform
                input_stereo = dataset[j,:,:].cuda()
                #stack mono version on top of stereo waveform (0 and 1 are stereo, 2 is mono)
                input_wav = torch.mean(input_stereo, dim=0).unsqueeze(0)
                #run the data through the network
                output_wav = model(input_wav.unsqueeze(0)).cpu()
                torchaudio.save(results_path + os.sep + "intermediate_output_" + str(counter) + "_" + str(j) + ".wav", output_wav.squeeze(),fs)

    print("Epoch {} finished! Average Loss: {}, Total Time: {}".format(counter,mean(epoch_losses),time.time()-start_time))
    #test to see if we should run another epoch
    if counter > 20:
        #if mean(training_losses[-4:]) < mean(epoch_losses):
        keep_training = False
        print("training finished!")

    training_losses.append(mean(epoch_losses))
    counter = counter + 1

print("saving results")
#generate results

with torch.no_grad():
    for j in range(dataset.size()[0]):
        #file,channel,waveform
        input_stereo = dataset[j,:,:].cuda()
        #stack mono version on top of stereo waveform (0 and 1 are stereo, 2 is mono)
        input_wav = torch.mean(input_stereo, dim=0).unsqueeze(0)

        #run the data through the network
        output_wav = model(input_wav.unsqueeze(0)).cpu()

        torchaudio.save(results_path + os.sep + "output_" + str(j) + ".wav", output_wav.squeeze(),fs)

#save state_dicts
print("results generated, saving model")

torch.save(model.state_dict(), 'model_param_acappella.pth')
