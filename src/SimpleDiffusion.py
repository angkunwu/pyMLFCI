# simplest diffusion
# based on https://e-dorigatti.github.io/math/deep%20learning/2023/06/25/diffusion.html
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm

data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1,2])), # mixture components  
    torch.distributions.Normal(torch.tensor([-4.,4.]),torch.tensor([1.,1.])) # components' distributions
)
dataset = data_distribution.sample(torch.Size([1000,1]))
sns.histplot(dataset[:,0])
plt.show()

TIME_STEPS = 250
BETA = 0.02

def do_diffusion(data,steps=TIME_STEPS,beta=BETA):
    """
    Perform diffusion on the input.
    """
    distributions, samples = [None], [data]
    xt = data
    for t in range(steps):
        q = torch.distributions.Normal(
            np.sqrt(1-beta)*xt, # mean of the distribution at time t (sqrt(1-beta) * xt
            np.sqrt(beta) # standard deviation of the distribution at time t (sqrt(beta)
        )
        xt = q.sample()
        distributions.append(q) # store the distribution at time t
        samples.append(xt) # store the sample at time t
    return distributions, samples

_, samples = do_diffusion(dataset)

for t in torch.stack(samples)[:,:,0].T[:100]:
    plt.plot(t,c='navy',alpha=0.1) # plot each sample in the diffusion process
plt.xlabel('diffusion time')
plt.ylabel('value')
plt.show()

# 2 NN models to predict the mean and variance of the distribution at each time step
# input to the models will be the current sample xt and the normalized time step t (0 to 1)
mean_model = torch.nn.Sequential(
    torch.nn.Linear(2,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,1) # output layer to predict the mean of the distribution
)
var_model = torch.nn.Sequential(
    torch.nn.Linear(2,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,1), # output layer to predict the variance of the distribution
    torch.nn.Softplus() # ensure variance is positive
)

def compute_loss(forward_distributions, forward_samples,mean_model,var_model):
    p = torch.distributions.Normal(
        torch.zeros(forward_samples[0].shape),
        torch.ones(forward_samples[0].shape)
    )# prior, N(0,1)
    loss = -p.log_prob(forward_samples[-1]).mean() # loss from the final sample to the prior

    for t in range(1,len(forward_samples)):
        xt = forward_samples[t]
        xprev = forward_samples[t-1]
        q = forward_distributions[t] #q(xt|x_{t-1})
        # normalize t between 0,1 and add it as
        # a new column to the inputs of the mu and sigma networks
        xin = torch.cat(
            (xt, (t/len(forward_samples))*torch.ones(xt.shape[0],1)),
            dim=1
        )# concatenate xt with time step to pass to the mean and variance model
        mu = mean_model(xin)
        sigma = var_model(xin)
        p = torch.distributions.Normal(mu,sigma)

        loss -= torch.mean(p.log_prob(xprev)) # loss from the predicted x_{t-1} to the actual x_{t-1}
        loss += torch.mean(q.log_prob(xprev)) # add the loss from the q distribution to ensure it matches the forward process
    return loss/len(forward_samples) # return the average loss over all time steps

optim = torch.optim.AdamW(
    itertools.chain(mean_model.parameters(),var_model.parameters()),
    lr=1e-2,weight_decay=1e-6,
)

loss_history = []
for e in tqdm(range(1000)):
    # every training have new samples from the diffusion process
    forward_distributions,forward_samples = do_diffusion(dataset) # perform diffusion to get the forward process
    optim.zero_grad() # zero the gradients
    loss = compute_loss(
        forward_distributions,
        forward_samples,
        mean_model,
        var_model
    ) # compute the loss for the current epoch
    loss.backward() # backpropagate the loss
    optim.step() # update the model parameters
    loss_history.append(loss.item()) # store the loss for plotting

plt.plot(loss_history)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Training steps')
plt.show()

def sample_reverse(mean_model,var_model,count,steps=TIME_STEPS):
    p = torch.distributions.Normal(
        torch.zeros((count,1)), # prior mean
        torch.ones((count,1)) # prior std
    ) # prior distribution N(0,1)
    xt = p.sample() # initial random state
    sample_history = [xt]
    for t in range(steps,0,-1):
        xin = torch.cat((xt,t*torch.ones(xt.shape)/steps),dim=1)
        p = torch.distributions.Normal(
            mean_model(xin), # predicted mean from the mean model
            var_model(xin) # predicted variance from the variance model
        )
        xt = p.sample()
        sample_history.append(xt)
    return sample_history

samps = torch.stack(sample_reverse(mean_model,var_model,1000))

for t in samps[:,:,0].T[:200]:
    plt.plot(t,c='C%d' % int(t[-1]>0),alpha=0.1) # plot each sample in the reverse diffusion process'
plt.xlabel('generation time')
plt.ylabel('value')
plt.show()

sns.histplot(samps[-1, :, 0])
plt.show()