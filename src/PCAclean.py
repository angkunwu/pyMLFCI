import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import utils
from src import FormFactorFuns as FFFs

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
S = (S + S.conj().T)/2 # make sure S is Hermitian
#eigenvalues, eigenvectors = np.linalg.eig(S)
eigenvalues, eigenvectors = np.linalg.eigh(S) # use eigh for Hermitian matrix
# sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = np.abs(eigenvalues[idx])
eigenvectors = eigenvectors[:,idx]

# convert components from Nx1 to Dx1
normalvectors = np.zeros(XT.shape,dtype=complex)
for k in range(Samples):
    normalvectors[:,k] = np.dot(XT,eigenvectors[:,k])/np.sqrt(eigenvalues[k]*Samples)

def obtainExpansion(XT,normalvectors):
    Samples = XT.shape[1]
    Expansion = np.zeros((Samples,Samples),dtype=complex)
    for k in range(Samples):
        Expansion[:,k] = np.dot(XT.conj().T,normalvectors[:,k])
    return Expansion

expansions = obtainExpansion(XT,normalvectors)

# generate a function that convert the 1D data back to 2D data
def convert1Dto2D(data1d,N,NBZ):
    #data2d = np.zeros((N,N*NBZ),dtype=complex)
    data2d = np.zeros((N*NBZ,N),dtype=complex)
    for k in range(NBZ):
        #data2d[:,k*N:(k+1)*N] = data1d[k*N*N:(k+1)*N*N].reshape(N,N)
        data2d[(k*N):((k+1)*N),:] = data1d[(k*N*N):((k+1)*N*N)].reshape(N,N)
    # transpose the data2d
    data2d = data2d.T
    return data2d
Cherns = np.zeros(400) # Chern number of first 20 components
for k in tqdm(range(400)):
    Component2d = convert1Dto2D(normalvectors[:,k],N,NBZ)
    Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
    Cherns[k] = sum(Bcurs)/2/np.pi

# plot Chern number vs component index
plt.figure(figsize=(6,4))
plt.scatter(range(50), Cherns[:50])
plt.xlabel('component index', fontsize=12)
plt.ylabel('Chern number', fontsize=12)
plt.show()


# plot histogram of coefficients at first several orders
fig, ax = plt.subplots(3,3, figsize=(8,8))
for i in range(3):
    for j in range(3):
        img_idx = i*3+j
        ax[i,j].hist(np.real(expansions[:,img_idx]), bins=50)
        ax[i,j].set_xlim(-10,15)
        ax[i,j].set_title(r'$C=$' + f'{Cherns[img_idx]:.4f}',fontsize=10)
ax[2,0].set_xlabel(r'amplitude Re($\mathbf{u}_j\mathbf{x}_n$)',fontsize=12)
ax[2,0].set_ylabel(r'# of samples',fontsize=12)
plt.show()
#plt.savefig('../Desktop/fig2.pdf', bbox_inches='tight')


sum(expansions[0,:] * Cherns)

test = expansions[0,0:4] * Cherns[:4]