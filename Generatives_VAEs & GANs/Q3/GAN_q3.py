# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:41:19 2019

@author: karm2204
"""

"""
References:
"""
#%%

# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/2
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

#%%

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import dataset


#%%

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

image_transform = transforms.Compose([
    transforms.ToTensor()
])



#%%     
class Generator(nn.Module):
    """ Generator. Input is noise and latent variables, output is a generated
    image.
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64 * 2),
            nn.ELU(),
            nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1, bias = False)
        )

        self.activation = nn.Sigmoid()

    def forward(self, input_):
        input_ = input_.view(input_.size(0), -1, 1, 1)
        input_ = self.main(input_)
        input_ = self.activation(input_)
        return input_

#%%
class Discriminator(nn.Module):
    """ Discriminator. Input is an image (real or generated), output is
    P(generated), continuous latent variables, discrete latent variables.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_ = self.main(image)
        input_ = input_.view(-1, 1).squeeze(1)
        return input_

#%%
class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 100
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lambda_gp = 10.0
#%%   
def compute_gradient_penalty(x, y, G):
    '''
        Random weight term for interpolation between real and fake samples
        Get random interpolation between real x and fake y samples
    '''
    alpha = torch.rand((x.size(0), 1, 1, 1), device = x.device)
    lin_interpol = alpha * x + (1-alpha) * y
    lin_interpol.requires_grad_(True)
    # need a fake grad output
    output = G.discriminator(lin_interpol)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=lin_interpol,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0]
    gradient = gradients.view(gradients.size(0), -1)
    norm_2 = gradient.norm(p=2, dim=1)
    gradient_penalty = ((norm_2 - 1).pow(2)).mean()
    return gradient_penalty
#%% 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html 

def visual_samples(G, dimensions, device, svhn_loader, step=0):
    # Generate new images
    z = torch.randn(64, dimensions, device=device)
    generated = G.generator(z)
    #debug
    torchvision.utils.save_image(generated, 'images/gan/gan-gen.png', normalize=False)
    #torchvision.utils.save_image(generated, f"images/gan/3_1gan-generated-{step}.png", normalize=False)
def disentangled_representation(G, dimensions, device, epsilon = 3):
    #Sample from prior p(z) which is a Std Normal
    z = torch.randn(dimensions, device=device)
    
    #Copy this tensor times its number of dimensions and make perturbations on each dimension
    #The first element is the original sample
    z = z.repeat(dimensions+1, 1)
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    generated = G.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/plus_eps.png', normalize=False)

    #Do the same with the negative epsilon
    epsilon = -2*epsilon
    for i, sample in enumerate(z[1:]):
        sample[i] += epsilon

    #Make a batch of the pertubations and pass it through the generator
    generated = G.generator(z)
    torchvision.utils.save_image(generated, 'images/gan/nega_eps.png', normalize=False)
    
#%%
    
def interpolation(G, dimensions, device):
    # Interpolate in the latent space between z_0 and z_1
    z_0 = torch.randn(1,dimensions, device=device)
    z_1 = torch.randn(1,dimensions, device=device)
    z_a = torch.zeros([11,dimensions], device=device)

    for i in range(11):
        a = i/10
        z_a[i] = a*z_0 + (1-a)*z_1

    generated = G.generator(z_a)
    torchvision.utils.save_image(generated, 'images/GAN/latent.png', normalize = False)
    
    # Interpolate in the data space between x_0 and x_1
    x_0 = G.generator(z_0)
    x_1 = G.generator(z_1)
    x_a = torch.zeros(11,x_0.size()[1],x_0.size()[2],x_0.size()[3], device = device)

    for i in range(11):
        a = i/10
        x_a[i] = torch.lerp(x_0, x_1, a)

    torchvision.utils.save_image(x_a, 'images/GAN/data.png', normalize = False)


def save_images(img_dir: str):
    import os
    G = GAN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G.load_state_dict(torch.load('GAN_q#3_save.pth', map_location = device))
    G = G.to(device)
    G.eval()
    
    for p in G.parameters():
        p.requires_grad = False

    for i in range(10):
        print(i)
        latents = torch.randn(100, 100, device=device)
        images = G.generator(latents)
        os.makedirs(f"{img_dir}/img/", exist_ok=True)
        for j, image in enumerate(images):
            filename = f"{img_dir}/img/{i * 100 + j:03d}.png"
            torchvision.utils.save_image(image, filename, normalize=False)

#%%
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    G = GAN()
    G = G.to(device)
    G.train()

    gen_step = 5

    D_optimizer = torch.optim.Adam(G.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    G_optimizer = torch.optim.Adam(G.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    train, valid, test = get_data_loader("svhn", 64)
    
    try: 
        G.load_state_dict(torch.load('GAN_q#3_save.pth', map_location=device))
        print('----Using saved model----')

    except FileNotFoundError:
        for epoch in range(5):
            print(f"------- EPOCH {epoch} --------")

            running_loss_d = 0
            running_loss_g = 0
            
            for i, (img, _) in enumerate(train):
                G.train()

                # Training the discriminator
                D_optimizer.zero_grad()                
                img = img.to(device)
                latents = torch.randn([img.shape[0], G.latent_dim], device=device)
                fakes = G.generator(latents).detach()
                
                fakes_score = G.discriminator(fakes)
                fakes_score_mean = fakes_score.mean()
                fakes_score_mean.backward()

                reals_score = G.discriminator(img)
                reals_score_mean = -reals_score.mean()
                reals_score_mean.backward()
                loss = fakes_score_mean + reals_score_mean
            
                grad_penalty = G.lambda_gp * compute_gradient_penalty(img, fakes, G)
                grad_penalty.backward()
                loss += grad_penalty
                
                D_optimizer.step()
                running_loss_d += loss

                # training the generator
                if i % gen_step == 0:
                    G_optimizer.zero_grad()
                    latents = torch.randn([img.shape[0], G.latent_dim], device=device)
                    fakes = G.generator(latents)

                    fakes_score = G.discriminator(fakes)
                    fakes_score_mean = -fakes_score.mean()
                    fakes_score_mean.backward()

                    G_optimizer.step()
                    running_loss_g += fakes_score_mean
                    
                if(i%10 == 0):
                    visual_samples(G, 100, device, test)

                if i % 100 == 0:
                    print(f"Training example {i} / {len(train)}. DiscLoss: {running_loss_d:.2f}, GenLoss: {running_loss_g:.2f}")
                    running_loss_d = 0
                    running_loss_g = 0
        
        torch.save(G.state_dict(), 'GAN_q#3_save.pth')

    dimensions = 100
        
    G.eval()
    #3_1 Visual samples
    visual_samples(G, dimensions, device, test)

    #3_2 Disentangled representation
    disentangled_representation(G, dimensions, device, epsilon=10)

    #3_3 Interpolation
    interpolation(G, dimensions, device)

    img_dir = "images/GAN/fid"
    save_images(img_dir)
