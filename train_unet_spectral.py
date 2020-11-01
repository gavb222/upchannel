from network_unet_spectral import UNet
import torch
import torchaudio
import torch.nn as nn
import os
import time

def mean(list):
    return sum(list)/len(list)

#build dataset
dataset_path = "D:/songs_headphones/stereo-cut-16k"
n_files = 0
fs = 0
for subdir, dirs, files in os.walk(dataset_path):

    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".wav"):
            if n_files == 0:
                dataset,fs = torchaudio.load(filepath)
            elif n_files == 1:
                new_file,fs = torchaudio.load(filepath)
                dataset = torch.stack((dataset,new_file))
            else:
                new_file,fs = torchaudio.load(filepath)
                dataset = torch.cat((dataset,new_file.unsqueeze(0)))

            n_files = n_files + 1

print("finished loading: {} files loaded".format(n_files))

#setup the network (refer to the github page)
#network input/output (N,C,H,W) = (1,1,256,256) => (1,2,256,256)
model = UNet(num_classes=2, in_channels=1, depth=5, merge_mode='concat')
model.cuda()
model.train()
loss_fn = nn.MSELoss()
criterion = torch.optim.Adam(model.parameters(), lr = .0002, betas = (.5,.999))

#for each epoch:
keep_training = True
training_losses = []
counter = 1
n_fft = 1023

window_tensor = torch.hann_window(n_fft).cuda()

print("training start!")
while keep_training:
    epoch_losses = []
    start_time = time.time()
    #do TiFGAN transformations to data
    for j in range(dataset.size()[0]):
        #file,channel,waveform
        input_wav = dataset[j,:,:].cuda()
        #stack mono version on top of stereo waveform (0 and 1 are stereo, 2 is mono)
        input_wav = torch.cat((input_wav,torch.mean(input_wav, dim=0).unsqueeze(0)))

        model.zero_grad()

        #build the spectrograms here (this isn't going to work right, i need to find the right parameters) WAS ONESIDED = TRUE
        complex_mix = torch.stft(input_wav, n_fft = n_fft, hop_length = 128, window = window_tensor)
        complex_mix_pow = complex_mix.pow(2).sum(-1)
        complex_mix_mag = torch.sqrt(complex_mix_pow)

        #adapted from TiFGAN
        complex_mix_mag = complex_mix_mag/torch.max(complex_mix_mag)

        complex_mix_mag = torch.log(complex_mix_mag)

        #now clip log mag at -10 minimum
        complex_mix_mag[complex_mix_mag < -10] = -10

        #scale to range of tanh:
        #was /5 + 1
        complex_mix_mag = (complex_mix_mag/10) + 1

        #/TiFGAN
        complex_mix_mag_mono = complex_mix_mag[2,:,:]
        complex_mix_mag = complex_mix_mag[:-1]

        #run the data through the network
        magspec_out = model(complex_mix_mag_mono.unsqueeze(0).unsqueeze(0))
        #do L1 (L2?, MSE?, crossEntropy?) loss on the network output
        loss = loss_fn(magspec_out.squeeze(),complex_mix_mag.squeeze())
        loss.backward()
        criterion.step()

        epoch_losses.append(loss.item())


    print("Epoch {} finished! Average Loss: {}, Total Time: {}".format(counter,mean(epoch_losses),time.time()-start_time))
    #test to see if we should run another epoch
    if counter > 15:
        if mean(training_losses[-4:]) < mean(epoch_losses):
            keep_training = False
            print("training finished!")

    training_losses.append(mean(epoch_losses))
    counter = counter + 1

print("saving results")
#generate results
results_path = "output"
to_wav = torchaudio.transforms.GriffinLim(n_fft = n_fft, hop_length = 128, n_iter = 100)

with torch.no_grad():
    for j in range(dataset.size()[0]):
        #file,channel,waveform
        input_wav = dataset[j,:,:].cuda()
        #stack mono version on top of stereo waveform (0 and 1 are stereo, 2 is mono)
        input_wav = torch.cat((input_wav,torch.mean(input_wav, dim=0).unsqueeze(0)))

        #build the spectrograms here (this isn't going to work right, i need to find the right parameters) WAS ONESIDED = TRUE
        complex_mix = torch.stft(input_wav, n_fft = n_fft, hop_length = 128, window = window_tensor)
        complex_mix_pow = complex_mix.pow(2).sum(-1)
        complex_mix_mag = torch.sqrt(complex_mix_pow)

        #adapted from TiFGAN
        complex_mix_mag = complex_mix_mag/torch.max(complex_mix_mag)

        complex_mix_mag = torch.log(complex_mix_mag)

        #now clip log mag at -10 minimum
        complex_mix_mag[complex_mix_mag < -10] = -10

        #scale to range of tanh:
        #was /5 + 1
        complex_mix_mag = (complex_mix_mag/10) + 1

        #/TiFGAN
        complex_mix_mag_mono = complex_mix_mag[2,:,:]

        #run the data through the network
        magspec_out = model(complex_mix_mag_mono.unsqueeze(0).unsqueeze(0)).cpu()

        #was *5
        magspec_out = (magspec_out - 1)*10
        magspec_out = torch.exp(magspec_out)
        magspec_out = magspec_out * torch.max(complex_mix_mag.cpu())

        macspec_out = magspec_out.view(2,512,512).contiguous()

        #from tacotron: x => |x|^1.2
        magspec_out = torch.pow(torch.abs(magspec_out),1.2)

        wav_out = to_wav(magspec_out)

        torchaudio.save(results_path + os.sep + "output_" + str(j) + ".wav",wav_out.squeeze(),fs)

#save state_dicts
print("results generated, saving model")

torch.save(model.state_dict(), 'model_param.pth')
