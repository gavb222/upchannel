
from network import UNet, Discriminator
import torch
import torchaudio
import torch.nn as nn
import os
import time

n_fft = 1023
window_tensor = torch.hann_window(n_fft).cuda()

def mean(list):
    return sum(list)/len(list)

def perceptual_loss(output,target):
    complex_mix_output = torch.stft(output, n_fft = n_fft, hop_length = 128, window = window_tensor)
    complex_mix_pow_output = complex_mix_output.pow(2).sum(-1)
    complex_mix_mag_output = torch.sqrt(complex_mix_pow_output)

    complex_mix_target = torch.stft(target, n_fft = n_fft, hop_length = 128, window = window_tensor)
    complex_mix_pow_target = complex_mix_target.pow(2).sum(-1)
    complex_mix_mag_target = torch.sqrt(complex_mix_pow_target)

    comparison = nn.MSELoss()
    return comparison(complex_mix_mag_output,complex_mix_mag_target)

#build dataset
dataset_path = "D:/songs_headphones/stereo-cut-16k"
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

print("finished loading: {} files loaded, Total Time: {}".format(n_files, time.time()-start_time))

#network input/output (N,C,H,W) = (1,1,256,256) => (1,2,256,256)
G = UNet(1,2)
D = Discriminator(3,64,1)
G.cuda()
D.cuda()
G.train()
D.train()

BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

G_criterion = torch.optim.Adam(G.parameters(), lr = .0001, betas = (.5,.999))
D_criterion = torch.optim.Adam(D.parameters(), lr = .0001, betas = (.5,.999))

#for each epoch:
keep_training = True
training_losses = []
counter = 1

results_path = "output"

print("training start!")
while keep_training:
    G_losses = []
    D_losses = []
    start_time = time.time()
    #do TiFGAN transformations to data
    for j in range(dataset.size()[0]):
        #file,channel,waveform
        input_stereo = dataset[j,:,:].cuda()
        #torch.mean is torchaudio.downmix_mono
        input_wav = torch.mean(input_stereo, dim=0).unsqueeze(0)

        D_real_input = torch.cat((input_stereo,input_wav),dim=0).unsqueeze(0)
        D_real_decision = D(D_real_input).squeeze()
        real_ = torch.ones(D_real_decision.size()).cuda()
        D_real_loss = BCE_loss(D_real_decision, real_)

        generated_wav = G(input_wav.unsqueeze(0))

        D_fake_input = torch.cat((generated_wav.squeeze(),input_wav),dim=0).unsqueeze(0)
        D_fake_decision = D(D_fake_input).squeeze()
        fake_ = torch.zeros(D_fake_decision.size()).cuda()
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_criterion.step()

        #train the generator
        #probably dont have to do this part, but i love bet hedging
        generated_wav = G(input_wav.unsqueeze(0))
        D_fake_input = torch.cat((generated_wav.squeeze(),input_wav),dim=0).unsqueeze(0)
        D_fake_decision = D(D_fake_input).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        percep_loss = 100 * perceptual_loss(generated_wav.squeeze(), input_stereo.squeeze())

        G_loss = G_fake_loss + percep_loss
        G.zero_grad()
        G_loss.backward()
        G_criterion.step()

        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        if j%100 == 0:
            print("=", end = '')
    print("\n")
    print("Epoch {} finished! G Loss: {}, D Loss: {}, Total Time: {}s".format(counter,mean(G_losses),mean(D_losses),time.time()-start_time))
    #test to see if we should run another epoch

    if counter%5 == 1:
        with torch.no_grad():
            for j in range(0,dataset.size()[0],100):
                #file,channel,waveform
                input_stereo = dataset[j,:,:].cuda()
                #stack mono version on top of stereo waveform (0 and 1 are stereo, 2 is mono)
                input_wav = torch.mean(input_stereo, dim=0).unsqueeze(0)

                #run the data through the network
                output_wav = G(input_wav.unsqueeze(0)).cpu()

                torchaudio.save(results_path + os.sep + "intermediate_output_" + str(counter) + "_" + str(j) + ".wav", output_wav.squeeze(),fs)
    if counter > 25:
        #if mean(training_losses[-4:]) < mean(epoch_losses):
        keep_training = False
        print("training finished!")

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
        output_wav = G(input_wav.unsqueeze(0)).cpu()

        torchaudio.save(results_path + os.sep + "output_" + str(j) + ".wav", output_wav.squeeze(),fs)

#save state_dicts
print("results generated, saving model")

torch.save(G.state_dict(), 'g_param.pth')
torch.save(D.state_dict(), 'd_param.pth')
