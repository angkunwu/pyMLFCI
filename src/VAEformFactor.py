import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import utils
from src import FormFactorFuns as FFFs

Nx, Ny = 3, 5
train_data, y = utils.ReadAllData(Nx,Ny) # the train_data is a 2D array with each column as one form factor vector

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

alphas = np.linspace(0.78943,3.517548,100)
FormFactor1d = train_data[:,99]
N = Nx * Ny
NBZ = FormFactor1d.shape[0]/N/N
NBZ = int(NBZ)
FormFactor2d = convert1Dto2D(FormFactor1d,N,NBZ)

BZind = int(NBZ/2)+1 # choose the center BZ
FFmiddle = FormFactor2d[:,((BZind-1)*N):(BZind*N)]
FFmiddle_phase = np.angle(FFmiddle)

plt.figure()
plt.imshow(FFmiddle_phase, cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
#plt.yscale('log')
plt.show()

import torch
from torch.utils.data import DataLoader, Dataset

class FormFactorDataset(Dataset):
    def __init__(self, data, labels,Nx,Ny,NBZ, transform=None):
        """
        Args:
            data(numpy.ndarray): 2D array where each column is a flattened image.
            labels (numpy.ndarray): Labels for corresponding each image
            Nx (int): Number of rows in the BZ.
            Ny (int): Number of columns in the BZ.
            NBZ (int): Number of BZ.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.Nx = Nx
        self.Ny = Ny
        self.NBZ = NBZ

    def __len__(self):
        return self.data.shape[1] 
    
    def __getitem__(self, idx):
        data1d = self.data[:, idx]
        data_real = data1d.real
        data_imag = data1d.imag
        data_combined = np.concatenate((data_real, data_imag), axis=0)
        # Normalize the data (optional, depending on your model)
        data_combined = (data_combined) / 2.0  # Scale to [-0.5, 0.5]
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data_combined, dtype=torch.float32)
        # Apply any additional transformations
        if self.transform:
            data_tensor = self.transform(data_tensor)
        label = self.labels[idx]
        label_tensor = torch.tensor(bool(label), dtype=torch.bool)
        return data_tensor, label_tensor

FFdataset = FormFactorDataset(train_data, y, Nx, Ny, NBZ)

# Create a DataLoaders
batch_size = 10
FFdataloader = DataLoader(FFdataset, batch_size=batch_size, shuffle=False)

FFimageTensor = FFdataset[99]
# convert the data to 2D with complex numbers
FFimageNP = FFimageTensor[0].numpy()*2.0
FFimagComplex = convert1Dto2D(FFimageNP[:(N*N*NBZ)],N,NBZ) + 1j*convert1Dto2D(FFimageNP[(N*N*NBZ):],N,NBZ)
FFmidtest = FFimagComplex[:,((BZind-1)*N):(BZind*N)]
FFmidtest_phase = np.angle(FFmidtest)
# check it FFmiddle_phase and FFmidtest_phase are the same
print(np.allclose(FFmiddle_phase, FFmidtest_phase))



from tqdm.auto import tqdm

writer = None #SummaryWriter(f'runs/mnist/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
# hyperparameters
batch_size = 10
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 10 # 50
latent_dim = 2
hidden_dim = 1024 # 512

from src import VAEclass

model = VAEclass.VAE(input_dim=2*N*N*NBZ,hidden_dim=hidden_dim,latent_dim=latent_dim).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params:,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

prev_updates = 0    
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    prev_updates = VAEclass.train(model,FFdataloader,optimizer,prev_updates,writer=writer,batch_size=batch_size)
    VAEclass.test(model,test_loader,prev_updates,writer=writer)


for batch_data, _ in FFdataloader:
    print("Batch Data Shape:", batch_data.shape)  # Should be (batch_size, flattened_size)
    break