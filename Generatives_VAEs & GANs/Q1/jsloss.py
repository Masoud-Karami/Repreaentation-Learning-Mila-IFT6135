from __future__ import print_function
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import samplers

#3-layer MLP
class Discriminator(nn.Module):
    
    def __init__(self,input_size=2,hidden_size=25,output_size=1):
        super(Discriminator, self).__init__()
        
        self.minibatch_size = 512
        self.lr = 1e-3
        self.epoch_count = 2
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
          
        return self.layers(x)
      
def JSD(x_val, minibatch_size, epoch_count, learning_rate, \
        input_size, hidden_size, output_size,  \
        real_sampler = samplers.distribution1, \
        fake_sampler = samplers.distribution1  \
        ):
    
    #initialize discriminator components
    detective = Discriminator().cuda()
    optimizer = optim.SGD(detective.parameters(), lr=learning_rate) 
    loss_fc = nn.BCELoss(reduction = 'mean').cuda()

    #get data for both distributions from samplers
    #the "real" distribution has value of phi instead of always 0
    real_dist = iter(real_sampler(x_val,512))
    fake_dist = iter(fake_sampler(0,512)) 
    
    real_targets = torch.ones([minibatch_size,1]).cuda()
    fake_targets = torch.zeros([minibatch_size,1]).cuda()

    for i in range(epoch_count):
        #get new samples
        real_samples = next(real_dist)
        real_tensor_samples = torch.tensor(real_samples).float().cuda()
        
        fake_samples = next(fake_dist)
        fake_tensor_samples = torch.tensor(fake_samples).float().cuda()

        real_output = detective(real_tensor_samples)
        fake_output = detective(fake_tensor_samples)

        #evaluate BCEloss
        loss_real_output = loss_fc(real_output, real_targets)
        loss_fake_output = loss_fc(fake_output, fake_targets)

        total_output = -loss_real_output/2 + -loss_fake_output/2 + \
                      torch.log(torch.tensor(2.).cuda())

        loss_real_output.backward()
        loss_fake_output.backward()
        optimizer.step()
    
    #after the training loop
    return total_output
