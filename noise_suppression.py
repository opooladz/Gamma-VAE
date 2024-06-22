#%%
import glob
import soundfile
import matplotlib.pyplot as plt
import scipy.signal as signal 
import numpy as np
import torch
import torchaudio
import time
from torch import nn, optim
device = torch.device('cuda')

#%% read wav files 
# wav_files = glob.glob('C:/Users/omead/Documents/VCTK-Corpus-0.92/wav48_silence_trimmed/**/*.flac', recursive=True)
wav_files = glob.glob('C:/Users/omead/Documents/subset_raw/raw/raw*.wav')
wavs_ = []
for wav_file in wav_files:
    # if np.random.rand() < 0.10:
    wav, fs = soundfile.read(wav_file)
    assert len(wav.shape) == 1
    wavs_.append(signal.resample_poly(wav, 16000, fs))
        
wavs_ = np.concatenate(wavs_, 0)





noise_files1 = glob.glob('C:\\Users\\omead\\Documents\\xilin_noise\\*.wav')
noise_files2 = glob.glob('C:/Users/omead/Documents/noise/*.wav')
noise_files = noise_files1 +  noise_files2
noise = []
for wav_file in noise_files:
    # if np.random.rand() < 0.10:
    wav, fs = soundfile.read(wav_file)

    noise.append(signal.resample_poly(wav[:,0], 16000, fs))
        
noise = np.concatenate(noise, 0)


wavs = wavs_
#%% stft
X = torchaudio.functional.spectrogram(
    waveform=torch.tensor(wavs, dtype=torch.float32),
    pad=0,
    window=torch.hann_window(512,periodic=True)/torch.sqrt(torch.tensor(1.5)),
    n_fft=512,
    hop_length=128,
    win_length=512,
    power=None,
    normalized=False)
X = X[1:-1, ::10]
X = X.t().to(device)
N = torchaudio.functional.spectrogram(
    waveform=torch.tensor(noise, dtype=torch.float32),
    pad=0,
    window=torch.hann_window(512,periodic=True)/torch.sqrt(torch.tensor(1.5)),
    n_fft=512,
    hop_length=128,
    win_length=512,
    power=None,
    normalized=False)
N = N[1:-1, ::10]
N = N.t().to(device)
#%%
dim_z = 32
dim_f = 255
dim_w = 2048
#%%
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(3*dim_f, dim_w)
        torch.nn.init.normal_(self.fc1.weight.data,std = 0.03)  
        
                
        self.fc2 = nn.Linear(dim_w, dim_z+1)
        torch.nn.init.normal_(self.fc2.weight.data,std = 0.02)     
        self.verbose = True 
        
    def forward(self, X):
        X = torch.stack([torch.real(X), torch.imag(X)], dim=-1)
        pwr = torch.sum(X*X, dim=-1, keepdim=True) + 1e-10
        log_pwr = torch.log(pwr)/10
        phase = 2*X/torch.sqrt(pwr)
        if self.verbose:
            print(torch.mean(phase*phase))
            print(torch.mean(log_pwr*log_pwr))
        
        x = torch.cat([log_pwr, phase], dim=-1)
        
        if self.verbose:
            print(x.shape)
        x = torch.reshape(x, [-1, dim_f*3])
        x = self.fc1(x)
        if self.verbose:
            print('after fcn  1')
            print(torch.mean(x*x))
        x = torch.tanh(x)
        
        x = self.fc2(x)
        if self.verbose:
            print('after fcn  2')
            print(torch.mean(x*x))
            print(x.shape)
            
        self.verbose = False
            
        mu = x[:, :dim_z]
        sigma = torch.exp(x[:, dim_z:]) 
        return [mu, sigma]
        
        

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc3 = nn.Linear(dim_z, dim_w)
        torch.nn.init.normal_(self.fc3.weight.data,std = 0.05) 
        
        self.fc4 = nn.Linear(dim_w,dim_f*3)      
        torch.nn.init.normal_(self.fc4.weight.data,std = 0.05)         
        self.verbose = True 
        
    def forward(self, x):
        if self.verbose:
            print(x.shape)
        
        x = self.fc3(x)
        if self.verbose:
            print('after fcn  3')
            print(torch.mean(x*x))
        x = torch.tanh(x)
        
        x = self.fc4(x)
        if self.verbose:
            print('after fcn  4')
            print(torch.mean(x*x))
            print(x.shape)
            
        Xhat = torch.exp(torch.complex(x[:,:dim_f], x[:, dim_f:dim_f*2]))
        sigma1 = torch.exp(x[:, dim_f*2:dim_f*3])
           
        self.verbose = False
        
        return [Xhat, sigma1]
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc3 = nn.Linear(dim_z, dim_w)
        torch.nn.init.normal_(self.fc3.weight.data,std = 0.05) 
        
        self.fc4 = nn.Linear(dim_w,dim_f*3*2)      
        torch.nn.init.normal_(self.fc4.weight.data,std = 0.05)         
        self.verbose = True 
        
    def forward(self, x):
        if self.verbose:
            print(x.shape)
        
        x = self.fc3(x)
        if self.verbose:
            print('after fcn  3')
            print(torch.mean(x*x))
        x = torch.tanh(x)
        
        x = self.fc4(x)
        if self.verbose:
            print('after fcn  4')
            print(torch.mean(x*x))
            print(x.shape)
            
        Xhat = torch.exp(torch.complex(x[:,:dim_f], x[:, dim_f:dim_f*2]))
        sigma1 = torch.exp(x[:, dim_f*2:dim_f*3])
        
        Nhat = torch.exp(torch.complex(x[:,dim_f*3:dim_f*4], x[:, dim_f*4:dim_f*5]))
        sigma2 = torch.exp(x[:, dim_f*5:dim_f*6])        
        self.verbose = False
        
        return [Xhat, sigma1, Nhat, sigma2]

#%%
def IS(X,Xhat,alpha):
    eps = 1e-8
    x = (torch.abs(X)+eps)
    y = (torch.abs(Xhat)+eps)
    return  x/y  - alpha*torch.log(x/y) + torch.lgamma(alpha) + 2.*torch.log(x)
#%% vae loss
def vae_loss(encoder, decoder, X, N):
    verbose=False
    mu, sigma = encoder(X)
    
    loss_kl = -0.5 + mu*mu/2 + sigma*sigma/2 - torch.log(sigma)
    if verbose:
        print(loss_kl.shape)
    
    # draw z
    z = mu + sigma*torch.randn_like(mu)
    
    # decoding 
    Xhat, sigma1, Nhat, sigma2 = decoder(z)
    

    E = X - Xhat
    E2 = N - Nhat
         
    loss_noisy_speech =  torch.log(torch.tensor(2.0*torch.pi)) + 2*torch.log(sigma1) + 0.5*torch.real(E*torch.conj(E))/(sigma1**2)
    loss_noise =   torch.log(torch.tensor(2.0*torch.pi)) + 2*torch.log(sigma2) + 0.5*torch.real(E2*torch.conj(E2))/(sigma2**2)
       
    # Shat = sigma1 - sigma2 
    
    # S = X - N 
    # an attempt at directly estimating weiner filter via VAE. Did not work that well
    # weiner = torch.abs(X)*torch.abs(Shat)**2/(torch.abs(Shat)**2+torch.abs(sigma2)**2)          


    # loss_speech = IS(S,weiner,alpha=torch.tensor(1,device='cuda:0'))
    
    
    
    loss_kl = torch.sum(loss_kl)/len(X)/255

    loss_noisy_speech = torch.sum(loss_noisy_speech)/len(X)/255
    loss_noise = torch.sum(loss_noise)/len(X)/255        
    loss_speech = torch.sum(loss_noisy_speech)/len(X)/255       

    loss_per_bin = loss_kl + loss_noisy_speech +  loss_noise #+  loss_speech#+ consistancy_loss

    
    return loss_per_bin, loss_noisy_speech, loss_noise, loss_speech

def vae_loss_weiner(encoder, decoder, X, N):
    verbose=False
    mu, sigma = encoder(X)
    
    loss_kl = -0.5 + mu*mu/2 + sigma*sigma/2 - torch.log(sigma)
    if verbose:
        print(loss_kl.shape)
    
    # draw z
    z = mu + sigma*torch.randn_like(mu)
    
    # decoding 
    Xhat, sigma1, Nhat, sigma2 = decoder(z)
    # Xhat, sigma1 = decoder(z)
    

    



    E = X - Xhat
    E2 = N - Nhat
         
    loss_noisy_speech =  torch.log(torch.tensor(2.0*torch.pi)) + 2*torch.log(sigma1) + 0.5*torch.real(E*torch.conj(E))/(sigma1**2)
    loss_noise =   torch.log(torch.tensor(2.0*torch.pi)) + 2*torch.log(sigma2) + 0.5*torch.real(E2*torch.conj(E2))/(sigma2**2)
       
    Shat = torch.abs(sigma1 - sigma2) 
    
    S = X - N 
    # weiner = torch.abs(X)*torch.abs(Shat)**2/(torch.abs(Shat)**2+torch.abs(sigma2)**2)          

    # x = (torch.abs(S+N)+eps)
    # y = (torch.abs(sigma1+sigma2)+eps)
    # consistancy_loss = x /y  - torch.log(x /y ) - 1    
    # loss_speech = IS(S,weiner,alpha=torch.tensor(1,device='cuda:0'))
    consistancy_loss = IS(S,Shat.detach(),alpha=torch.tensor(1,device='cuda:0'))

    
    
    loss_kl = torch.sum(loss_kl)/len(X)/255

    loss_noisy_speech = torch.sum(loss_noisy_speech)/len(X)/255
    loss_noise = torch.sum(loss_noise)/len(X)/255        
    # loss_speech = torch.sum(loss_speech)/len(X)/255      
    consistancy_loss = torch.sum(consistancy_loss)/len(X)/255 

    loss_per_bin = loss_kl + loss_noisy_speech +  loss_noise + consistancy_loss #+  loss_speech#+ consistancy_loss

    
    return loss_per_bin, loss_noisy_speech, loss_noise, consistancy_loss

 #%%
from adabelief_pytorch import AdaBelief



last_total_loss = torch.inf
lr = 1e-4
# X_window = X_window.to(device, dtype=torch.cfloat)
# X_window = X_window.permute(0,2,1)
# start_training_time = time.time()
# Loss = []

encoder, decoder = Encoder().to(device), Decoder().to(device)
# encoder = torch.load(f'model_multi_batched_CN_bigger_nw_long_init_adab_e-3_fcn_normal_enc_hop_128_run0_full_noise_and_speech_minus.mdl')
# decoder = torch.load( f'model_multi_batched_CN_bigger_nw_long_init_adab_e-3_fcn_normal_dec_hop_128_run0_full_noise_and_speech_minus.mdl')

codec_params = list(decoder.parameters()) + list(encoder.parameters())
import numpy as np
import torch
from torch.optim.optimizer import Optimizer



#%%
# optimizer = AdaBelief(codec_params, lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decouple = False, rectify = False)
batch = 128

optimizer = AdaBelief(
    codec_params,
    lr=1e-3,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decouple=False,
    rectify=False,
)
# optimizer = optim.Adam(codec_params,lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[300, 350, 450], gamma=0.5
)
last_total_loss = torch.inf    
n_len = N.shape[0]
Losses = []

#%%
for _ in range(500):
    # t0 = time.time()
    # X = X[torch.randperm(len(X))]
    rand_perm = torch.randperm(len(X))
    i = 0
    this_total_loss = 0.0
    speech_total_loss = 0.0
    noise_total_loss = 0.0
    noisy_speech_total_loss = 0.0
    consistancy_total_loss = 0.0
    while i + batch <= len(X):

        start_idx = np.random.randint(0,n_len-batch)

        optimizer.zero_grad()
        if _ < 50:
            location = 10; var = 5
        elif _ > 50 and _ < 100:
            location = 8; var = 5
        elif _ > 100 and _ < 150:
            location = 6; var = 4
        elif _ > 150 and _ < 200:
            location = 5; var = 3
        elif _ > 200 and _ < 250:
            location = 4; var = 3
        elif _ > 250 and _ < 300:
            location = 3; var = 2
        Noise = N[start_idx:start_idx+batch]/np.abs(np.random.normal(loc=location,scale=var)) 
        S = X[rand_perm[i:i+batch]] + Noise     
        if True:
        # if _ < 100:                                                      
            loss, loss_x, loss_n , loss_s   = vae_loss(encoder, decoder, S  ,Noise)
        elif False:
            # Try to explcitly use the Weiner filter style 
            # May or may not work give the idea a try. 
            loss, loss_x, loss_n , loss_s   = vae_loss_weiner(encoder, decoder, S  ,Noise)
        loss.backward()
        # print(loss.item())
        this_total_loss += loss.item()
        speech_total_loss += loss_s.item()
        noise_total_loss += loss_n.item()
        noisy_speech_total_loss += loss_x.item()
        Losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(encoder.parameters()), 1., norm_type=2)
        optimizer.step()
            
        i += batch
    scheduler.step()

    last_total_loss = this_total_loss
    print('Epoch: {}; avg loss: {}; x loss {}; n loss {}; s loss {}'.format(_,this_total_loss/(len(X)//batch),noisy_speech_total_loss/(len(X)//batch),noise_total_loss/(len(X)//batch),speech_total_loss/(len(X)//batch)))

        
plt.plot(Losses)
plt.show()

#%%


S_ = torchaudio.functional.spectrogram(
    waveform=torch.tensor(wavs[:160000*2], dtype=torch.float32),
    pad=0,
    window=torch.hann_window(512)/torch.sqrt(torch.tensor(1.5)),
    n_fft=512,
    hop_length=128,
    win_length=512,
    power=None,
    normalized=False)
S_ = S_[1:-1]
S_ = S_.t().to(device)

# mu, sigma = encoder(S_+N[:S_.shape[0],:]/0.5)
# _, Shat, _, Nhat = decoder(mu)
# S_noisy = S_+N[:S_.shape[0],:]/0.5
#%%
S_noisy = S_+N[:S_.shape[0],:]/0.5
mu, sigma = encoder(S_noisy)
_1, Xhat, _2, Nhat = decoder(mu)
Shat = Xhat - Nhat 
Shat2 = S_noisy - Nhat
weiner = torch.abs(S_noisy)*torch.abs(Shat)**2/(torch.abs(Shat)**2+torch.abs(Nhat)**2)
weiner2 = torch.abs(S_noisy)*torch.abs(Shat2)**2/(torch.abs(Shat2)**2+torch.abs(Nhat)**2)
weiner3 = torch.abs(S_noisy)*torch.abs(Shat2)**2/(torch.abs(Shat)**2+torch.abs(Nhat)**2)

#%%
# plt.imshow(torch.log(torch.abs(Xhat)).cpu().detach().numpy().T[::-1])
plt.imshow(torch.log(torch.abs(S_noisy)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(S_)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(Xhat)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(Shat)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(weiner)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
# plt.imshow(torch.log(torch.abs(weiner2)).cpu().detach().numpy().T[::-1][:,:129*4])
# plt.show()
# plt.imshow(torch.log(torch.abs(weiner3)).cpu().detach().numpy().T[::-1][:,:129*4])
# plt.show()
#%%
# plt.imshow(torch.log(torch.abs(N[:X_.shape[0],:]/15)).cpu().detach().numpy().T[::-1][:,:129*4])
# plt.show()
# plt.imshow(torch.log(torch.abs(Xhat)).cpu().detach().numpy().T[::-1])
plt.imshow(torch.log(torch.abs(X_+N_[:X_.shape[0],:]/10)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(X_)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(Xhat)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(Nhat)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(N_[:X_.shape[0],:]/2)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()
plt.imshow(torch.log(torch.abs(Xhat-Nhat)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()

plt.imshow(torch.log(torch.abs(Xhat-Nhat)).cpu().detach().numpy().T[::-1][:,:129*4])
plt.show()


# plt.imshow(torch.log(torch.abs(_1)).cpu().detach().numpy().T[::-1][:,:129*4])
# plt.show()
# plt.imshow(torch.log(torch.abs(_2+_1)).cpu().detach().numpy().T[::-1][:,:129*4])
# plt.show()
# %%
torch.save(encoder, f'enc_fc_gamma_x_n_low_SNR.mdl')
torch.save(decoder, f'dec_fc_gamma_x_n_low_SNR.mdl')
