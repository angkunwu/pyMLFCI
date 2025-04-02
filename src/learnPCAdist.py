import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import utils
from src import FormFactorFuns as FFFs

def PCAcoefToPhase(coef):
    coefcomplex = coef[:Samples] + 1j*coef[Samples:]
    FFres = np.zeros((N*N*NBZ),dtype=complex)
    for k in range(Samples):
        FFres += coefcomplex[k]*normalvectors[:,k]
    FFres += barx # very important to add the mean!!!
    FF2d = utils.convert1Dto2D(FFres,N,NBZ)
    # Get the phase of the complex data
    BZind = int(NBZ/2)+1 # choose the center BZ
    data_phase = np.angle(FF2d[:,((BZind-1)*N):(BZind*N)])
    return data_phase


Nx, Ny = 3, 5
N = Nx*Ny
train_data, y = utils.ReadAllData(Nx,Ny)
NBZ = int(train_data.shape[0]/N/N)

Samples = np.size(y)
# mean of the training data
barx = np.mean(train_data,axis=1)
# Principal component analysis
XT = train_data - barx[:,None]
X = XT.conj().T
S = np.dot(X,XT)/Samples # N by N matrix
eigenvalues, eigenvectors = np.linalg.eig(S)
# convert components from Nx1 to Dx1
normalvectors = np.zeros(XT.shape,dtype=complex)
for k in range(Samples):
    normalvectors[:,k] = np.dot(XT,eigenvectors[:,k])/np.sqrt(eigenvalues[k]*Samples)

def obtainExpansion(XT,normalvectors):
    Samples = XT.shape[1]
    Expansion = np.zeros((Samples,Samples),dtype=complex)
    for k in tqdm(range(Samples)):
        Expansion[:,k] = np.dot(XT.conj().T,normalvectors[:,k])
    return Expansion

expansions = obtainExpansion(XT,normalvectors) # each row is the expansion coefficients of one form factor

# use Gaussian mixture model to learn the distribution of expansion coefficients
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
pca_coefs = np.zeros((Samples, 2*Samples), dtype=float)
for k in range(Samples):
    pca_coefs[k, :] = np.concatenate((np.real(expansions[k,:]), np.imag(expansions[k,:])), axis=0)
gmm.fit(pca_coefs)
samples = gmm.sample(10)

# plot the sample coefficients
fig, ax = plt.subplots(3,3, figsize=(8,8))
for i in range(3):
    for j in range(3):
        img_idx = i*3+j
        ax[i,j].scatter(np.arange(1,2*Samples+1),np.abs(samples[0][img_idx]))
        ax[i,j].set_xlim(0,250)
        ax[i,j].set_yscale('log')
plt.show()


fig, ax = plt.subplots(3,3, figsize=(8,8))
for i in range(3):
    for j in range(3):
        img_idx = i*3+j
        im = ax[i,j].imshow(PCAcoefToPhase(samples[0][img_idx,:]),cmap='RdBu', interpolation='nearest')
        #ax[i,j].axis('off')
        #fig.colorbar(im, ax=ax[i, j], orientation='vertical')
plt.show()

import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.transforms import AffineCouplingTransform, ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.flows.base import Flow

class TransformNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__() # call the parent constructor
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, inputs, context=None):
        return self.net(inputs)

pca_coefs = np.zeros((Samples, 2*Samples), dtype=float)
for k in range(Samples):
    pca_coefs[k, :] = np.concatenate((np.real(expansions[k,:]), np.imag(expansions[k,:])), axis=0)
pca_coefs_tensor = torch.tensor(pca_coefs, dtype=torch.float32)

base_distribution = StandardNormal(shape=(2*Samples,))
# define the transformation with stacking multiple coupling layers
num_layers = 5
hidden_dim = 128  # Hidden layer size
transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2*Samples)) # reverse permutation
    transforms.append(AffineCouplingTransform(
        mask=torch.arange(2*Samples) % 2,
        transform_net_create_fn=lambda in_features, out_features: TransformNet(
            input_dim=in_features,
            hidden_dim=hidden_dim,
            output_dim=out_features
        ),
    ))# affine coupling
transform = CompositeTransform(transforms) # composite transform

flow = Flow(transform, base_distribution)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
num_epochs = 10000
batch_size = 100
for epoch in tqdm(range(num_epochs)):
    indices = np.random.choice(Samples, batch_size, replace=False)
    batch = pca_coefs_tensor[indices]
    loss = -flow.log_prob(inputs=batch).mean()
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# sampling
num_samples = 10
samples = flow.sample(num_samples)
samples_np= samples.detach().numpy()

fig, ax = plt.subplots(3,3, figsize=(8,8))
for i in range(3):
    for j in range(3):
        img_idx = i*3+j
        im = ax[i,j].imshow(PCAcoefToPhase(samples_np[img_idx,:]),cmap='RdBu', interpolation='nearest')
        #ax[i,j].axis('off')
        #fig.colorbar(im, ax=ax[i, j], orientation='vertical')
plt.show()
