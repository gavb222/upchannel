
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

#build dataset
dataset_path = "D:/songs_headphones/acappella-cut"
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
G = UNet(num_classes=2, in_channels=1, depth=5, merge_mode='concat')
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
to_wav = torchaudio.transforms.GriffinLim(n_fft = n_fft, hop_length = 128, n_iter = 100)

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
        input_mono = torch.mean(input_stereo, dim=0).unsqueeze(0)

        #stack the mono signal on top of the stereo signal
        input_wav = torch.cat((input_stereo,input_mono))
        #TiFGAN

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
        complex_mix_mag_stereo = complex_mix_mag[:-1]

        #D_real_input = complex_mix_mag since its already catted
        #D_real_input = torch.cat((input_stereo,input_wav),dim=0).unsqueeze(0)
        D_real_decision = D(complex_mix_mag.unsqueeze(0)).squeeze()
        real_ = torch.ones(D_real_decision.size()).cuda()
        D_real_loss = BCE_loss(D_real_decision, real_)

        generated_spec = G(complex_mix_mag_mono.unsqueeze(0).unsqueeze(0))

        D_fake_input = torch.cat((generated_spec.squeeze(),complex_mix_mag_mono.unsqueeze(0)),dim=0).unsqueeze(0)
        D_fake_decision = D(D_fake_input).squeeze()
        fake_ = torch.zeros(D_fake_decision.size()).cuda()
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_criterion.step()

        #train the generator
        #probably dont have to do this part, but i love bet hedging
        generated_spec = G(complex_mix_mag_mono.unsqueeze(0).unsqueeze(0))
        D_fake_input = torch.cat((generated_spec.squeeze(),complex_mix_mag_mono.unsqueeze(0)),dim=0).unsqueeze(0)
        D_fake_decision = D(D_fake_input).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        percep_loss = 100 * L1_loss(generated_spec.squeeze(), complex_mix_mag_stereo.squeeze())

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
                magspec_out = G(complex_mix_mag_mono.unsqueeze(0).unsqueeze(0)).cpu()

                #was *5
                magspec_out = (magspec_out - 1)*10
                magspec_out = torch.exp(magspec_out)
                magspec_out = magspec_out * torch.max(complex_mix_mag.cpu())

                macspec_out = magspec_out.view(2,512,512).contiguous()

                #from tacotron: x => |x|^1.2
                magspec_out = torch.pow(torch.abs(magspec_out),1.2)

                wav_out = to_wav(magspec_out)

                torchaudio.save(results_path + os.sep + "intermediate_output_" + str(counter) + "_" + str(j) + ".wav",wav_out.squeeze(),fs)
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
        magspec_out = G(complex_mix_mag_mono.unsqueeze(0).unsqueeze(0)).cpu()

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

torch.save(G.state_dict(), 'g_param.pth')
torch.save(D.state_dict(), 'd_param.pth')
