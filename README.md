# Gamma-VAE
Reproduction of Gamma VAE from scratch -- I coded things up based on the paper 2 years later. 

I will try to clean it up more and solve LBFGS issue. For now there is a [google colab](https://colab.research.google.com/drive/14M0guLIVqk6CYrUvAFhiNBR8yu5wjwEx?authuser=5#scrollTo=LNLpustWq5ly) linked



Note Adam and SGD do not optimize speech well. We found PSGD does a really good job optimizing NNs for use with speech models. 

## Real Spectrogram:

![real](https://github.com/opooladz/Gamma-VAE/assets/16969636/87946b29-bc57-4d05-9750-822229a95a9b)

## Resynthesized Spectrogram:

![download](https://github.com/opooladz/Gamma-VAE/assets/16969636/dda25861-178b-4537-9142-15f64559e737)

note i did not decay lr of optimizer or do anything fancy. the results in the paper were better due to these adjustments but this will give a good base proof of concept. 
