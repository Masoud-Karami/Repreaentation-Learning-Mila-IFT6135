# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:05:29 2019

@author: karm2204
"""

"""
References:
"""
#%%

# https://towardsdatascience.com/model-summary-in-pytorch-b5a1e4b64d25
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://discuss.pytorch.org/t/text-autoencoder-nan-loss-after-first-batch/22730
# https://discuss.pytorch.org/t/understanding-output-padding-cnn-autoencoder-input-output-not-the-same/22743
# https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
# https://github.com/pytorch/examples/blob/master/vae/main.py
# https://www.groundai.com/project/isolating-sources-of-disentanglement-in-variational-autoencoders/
# https://arogozhnikov.github.io/einops/pytorch-examples.html

# Note   :    https://www.cs.toronto.edu/~lczhang/360/lec/w03/convnet.html
#%%

import argparse
import torch
import torch.nn as nn 
from torch import cuda
import torch.utils.data
from torch import optim, autograd
from torch.nn import functional as F
import torchvision
from torch.utils.data import dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt


batch_size = 64

#%%    
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])
    
def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    train = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    test = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return train, valid, test

#%%

def get_data_loader_1(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    train = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    test = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform_1
        ),
        batch_size=batch_size,
    )

    return train, valid, test

image_transform_1 = transforms.Compose([
    transforms.ToTensor()
])

#%%
def imshow(img):
    img = 0.5*(img + 1)
    npimg = img.numpy()
    # npimg = (255*npimg).astype(np.uint8) # to be a int in (0,...,255)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()                 
                    
#%% 
class View(nn.Module):
    def __init__(self, shape, *shape_):
        super().__init__()
        if isinstance(shape, list):
            self.shape = shape
        else:
            self.shape = (shape,) + shape_      
            
def forward(self, x):
        return x.view(self.shape)
                    
#%%                
# https://discuss.pytorch.org/t/text-autoencoder-nan-loss-after-first-batch/22730
                    
class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.convencoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            #  Layer 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # Layer 3
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # Layer 4
            View(self.batch_size, 4*4*512),
            nn.Linear(4*4*512, 2*self.latent_dim)
        )

        self.convdecoder = nn.Sequential(
            # Layer 1
            nn.Linear(self.latent_dim, 4*4*512),
            View(self.batch_size, 512, 4, 4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # Layer 2
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # Layer 3
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # Layer 4
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )


    def forward(self, x):
        convencoder = self.convencoder(x)
        mu, logvar = convencoder[:, :self.latent_dim], convencoder[:, self.latent_dim:]
        z = torch.randn_like(mu, device = args.device)
        x_hat = mu + torch.exp(logvar) * z
        decode_z = self.convdecoder(x_hat)
        return mu, logvar, decode_                                        
#%%

def ELBO(output, target, mu, logvar):
    elbo = -torch.nn.functional.mse_loss(output, target, reduction='sum')
    elbo += 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
    return elbo / output.size(0)

# https://www.groundai.com/project/isolating-sources-of-disentanglement-in-variational-autoencoders/
#%%
def visual_samples(vae, dimensions, device, svhn_loader):
    z = torch.randn(64, dimensions, device = device)
    generated = vae.convdecoder(z)
    torchvision.utils.save_image(generated, 'images/vae/3_1_VAE-generated.png', normalize=False)
    
#%%
def disentangled_representation(vae, dimensions, device, epsilon = 3):
    z = torch.randn(dimensions, device = device)
    z = z.repeat(dimensions+1, 1)
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    generated = vae.convdecoder(z)
    torchvision.utils.save_image(generated, 'images/vae/3_2positive_eps.png', normalize=False)
    epsilon = -2*epsilon
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    generated = vae.convdecoder(z)
    torchvision.utils.save_image(generated, 'images/vae/3_2negative_eps.png', normalize=False)

#%%
    
def interpolation(vae, dimensions, device):
    # Interpolate in the latent space between z_0 and z_1
    z_0 = torch.randn(1,dimensions, device=device)
    z_1 = torch.randn(1,dimensions, device=device)
    z_a = torch.zeros([11,dimensions], device=device)

    for i in range(11):
        a = i/10
        z_a[i] = a*z_0 + (1-a)*z_1

    generated = vae.convdecoder(z_a)
    torchvision.utils.save_image(generated, 'images/vae/3_3latent.png', normalize=False)
    
    # Interpolate in the data space between x_0 and x_1
    x_0 = vae.convdecoder(z_0)
    x_1 = vae.convdecoder(z_1)
    x_a = torch.zeros(11,x_0.size()[1],x_0.size()[2],x_0.size()[3], device = device)

    for i in range(11):
        a = i/10
        x_a[i] = a*x_0 + (1-a)*x_1

    torchvision.utils.save_image(x_a, 'images/vae/3_3data.png', normalize=False)   
    
    
#%%
def save_images(img_dir: str):
    import os
    vae = VAE()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.load_state_dict(torch.load('VAE_q#3_save.pth', map_location=device))
    vae = vae.to(device)
    vae.eval()
    
    for p in vae.parameters():
        p.requires_grad = False
        os.makedirs(f"{img_dir}/img/", exist_ok=True)
    for i in range(10):
        print(i)
        latents = torch.randn(100, 100, device=device)
        images = vae.decoder(latents)
        for j, image in enumerate(images):
            filename = f"images/vae/fid/img/{i * 100 + j:03d}.png"
            torchvision.utils.save_image(image, filename, normalize=True)
            
#%%        
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Let's use {}".format(device))
    vae = VAE()
    vae = vae.to(device)
    running_loss = 0
    optimizer = optim.Adam(vae.parameters(), lr=3e-4)
    train, valid, test = get_data_loader("svhn", batch_size = 64)
    try: 
        vae.load_state_dict(torch.load('VAE_q#3_save.pth', map_location=device))
        print('----Using saved model----')
    except FileNotFoundError:
        for epoch in range(20):
            print(f"------- EPOCH {epoch} --------")
            for i, (x, _) in enumerate(train):
                vae.train()
                optimizer.zero_grad()
                x = x.to(device)
                y, mu, logvar = vae(x)
                loss = -ELBO(y, x, mu, logvar)
                running_loss += loss
                loss.backward()
                optimizer.step()
                if(i%10 == 0):
                    visual_samples(vae, 100, device, test)

                if (i + 1) % 100 == 0:
                    print(f"Training example {i + 1} / {len(train)}. Loss: {running_loss}")
                    running_loss = 0

        torch.save(vae.state_dict(), 'VAE_q#3_save.pth')

    dimensions = 100
    
    
    visual_samples(vae, dimensions, device, test)
    disentangled_representation(vae, dimensions, device, epsilon=10)
    interpolation(vae, dimensions, device)
    
    img_dir = "images/vae/fid"
    save_images(img_dir)
