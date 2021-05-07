import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import samplers


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

#3-layer MLP
class Critic(nn.Module):
    
    def __init__(self,input_size=1,hidden_size=10,output_size=1):
        super(Critic, self).__init__()
        
        self.minibatch_size = 512
        self.lr = 1e-3
        self.epoch_count = 2
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, x):
          
        return self.layers(x)
      
def WGAN_GPloss(x_val, minibatch_size, epoch_count, learning_rate, losses,\
                input_size, hidden_size, output_size, penalty_coef = 10,\
                real_sampler=samplers.distribution1,\
                fake_sampler=samplers.distribution1,\
               ):
    
    critic = Critic(input_size, hidden_size, output_size).cuda()
    optimizer = optim.SGD(critic.parameters(),lr = learning_rate)

    #get both distributions from samplers
    #the "real" dist has value phi instead of always 0
    real_dist = iter(real_sampler(x_val,512))
    fake_dist = iter(fake_sampler(0,512)) 

################################gradient penalty################################
  
################################################################################
  
    for i in range(epoch_count):
        #get batch of data
        real_samples = next(real_dist)
        real_tensor_samples = torch.tensor(real_samples, requires_grad = True).float().cuda()

        fake_samples = next(fake_dist)
        fake_tensor_samples = torch.tensor(fake_samples, requires_grad = True).float().cuda()


        critic.zero_grad()
        optimizer.zero_grad()

        real_critic_output = critic(real_tensor_samples)
        fake_critic_output = critic(fake_tensor_samples)

        #make mix
        alpha = torch.rand([512,1]).cuda() #mixing coefficient, might need to be made 512 dim
        z = alpha*real_tensor_samples + (1-alpha)*fake_tensor_samples
        critic_z = critic(z)

        #compute gradient of critic(z) w.r.t. z
        grads = torch.autograd.grad(outputs = critic_z, \
                                    inputs = z, \
                                    grad_outputs = torch.ones([512,2]).cuda(), \
                                    only_inputs = True, \
                                    create_graph = True, \
                                    retain_graph = True
                                   )[0]

        #compute the full penalty (coef*mean of norm of grad)
        grad_penalty = ((torch.norm(grads, p=2, dim=1)-1)**2)

        real_critic_output_expected = real_critic_output.mean()
        fake_critic_output_expected = fake_critic_output.mean()
        grad_penalty_mean = grad_penalty.mean()

        total_output = -real_critic_output_expected + fake_critic_output_expected + \
                        penalty_coef*(grad_penalty_mean)

        total_output.backward()
        optimizer.step()

    return real_critic_output_expected - fake_critic_output_expected
