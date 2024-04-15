# Gamma-VAE
Reproduction of Gamma VAE from scratch -- I coded things up based on the paper 2 years later. 

I will try to clean it up more and solve LBFGS issue. For now there is a [google colab](https://colab.research.google.com/drive/14M0guLIVqk6CYrUvAFhiNBR8yu5wjwEx?authuser=5#scrollTo=LNLpustWq5ly) linked



Note Adam and SGD do not optimize speech well. We found PSGD does a really good job optimizing NNs for use with speech models. 

## Real Spectrogram:

![real](https://github.com/opooladz/Gamma-VAE/assets/16969636/87946b29-bc57-4d05-9750-822229a95a9b)

## Resynthesized Spectrogram:

![download](https://github.com/opooladz/Gamma-VAE/assets/16969636/dda25861-178b-4537-9142-15f64559e737) 


## architecure changes 

1D conv for encoder and 1D deconv for decoder will improve these results a bit

## (2024) Optimizing Griff Lim over STOI instead of SNR improves phase over what was in the origional paper 

Consider weights of model [here](https://drive.google.com/drive/folders/1PyBuKEeM6cwKvgQGJ7qELObzZ1teLgPj?usp=sharing)

You can also consider a snipit of [re-synthesized speech](https://drive.google.com/file/d/1-HL_5xAfeRmd7ugTAlMRh0hUP9OZGuOy/view?usp=drive_link)


