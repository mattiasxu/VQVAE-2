# VQVAE-2

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

VQVAE-2 https://arxiv.org/pdf/1906.00446.pdf

Autoregressive PixelSNAIL https://arxiv.org/pdf/1712.09763.pdf

Implementations based on https://github.com/rosinality/vq-vae-2-pytorch and https://github.com/EugenHotaj/pytorch-generative

## Step 1: Learning discrete latent space with VQVAE-2
To train on CIFAR10 with default parameters:
```
python train_vqvae.py
```
The VQVAE encoder compresses 3x32x32 images to 8x8 + 4x4 latent codes, which is a reduction of ~97%, using 8-bit color representations and codebook size of 512. The decoder tries to reconstruct the original image from latent codes.

Example reconstructions using default parameters from CIFAR10 test set. Original on top, reconstructions on bottom.

<img src="reconstruction_examples.png" width="768" height="128" />

## Step 2: Train autoregressive model on latent codes
First make dataset with latent codes encoded by the trained VQVAE-2.
```
python make_latent_dataset.py
```
Now, trian the autoregressive model
```
python train_pixelsnail.py
```

