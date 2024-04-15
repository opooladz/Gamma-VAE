# Gamma-VAE
Reproduction of Gamma VAE from scratch -- I coded things up based on the paper 2 years later. 

When we did the paper I did not try optimizing STOI as the objective for Griffin-Lim for finding a compatible phase.

I find this significantly increases the phase performance and likely matches or even beats the PESQ and FAD of the true phase. 

In addition at the time of the paper PSGD was not working well for solving Griffin-Lim. Now it is faster and performs better than LBFGS. 

Here is a [google colab](https://colab.research.google.com/drive/14M0guLIVqk6CYrUvAFhiNBR8yu5wjwEx?authuser=5#scrollTo=LNLpustWq5ly) that goes over everything.


Note Adam and SGD do not optimize speech well. We found PSGD does a really good job optimizing NNs for use with speech models. 

## Real Spectrogram:
![og_spec](https://github.com/opooladz/Gamma-VAE/assets/16969636/6069e216-43ac-49eb-a3d4-4440527f7eb7)


## Resynthesized Spectrogram:

![resynth_spec](https://github.com/opooladz/Gamma-VAE/assets/16969636/cfa37179-dabd-48a9-8c4e-615244f957e2)


## Architecure Changes 

1D conv for encoder and 1D deconv for decoder will improve these results a bit. I will try to update if I have time. 

## (2024) Optimizing Griff Lim over STOI instead of SNR improves phase over what was in the origional paper 

Consider weights of model [here](https://drive.google.com/drive/folders/1PyBuKEeM6cwKvgQGJ7qELObzZ1teLgPj?usp=sharing)

You can also consider a snipit of [re-synthesized speech](https://drive.google.com/file/d/19Hekb1j9RE5QgPE6J6TAVHreqAClzkta/view?usp=drive_link)
Given that this is the [origional speech](https://drive.google.com/file/d/17kK84dP9aiRN6e9Cab659HTkzHnECICO/view?usp=drive_link)

