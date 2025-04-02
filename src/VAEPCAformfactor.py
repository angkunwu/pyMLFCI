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

FormFactor2d = utils.convert1Dto2D(train_data[:,0],N,NBZ)
BZind = int(NBZ/2)+1 # choose the center BZ
FFmiddle = FormFactor2d[:,((BZind-1)*N):(BZind*N)]
FFmiddle_phase = np.angle(FFmiddle)

plt.figure()
plt.imshow(FFmiddle_phase, cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
#plt.yscale('log')
plt.show()



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


import torch
from torch.utils.data import DataLoader, Dataset

class FFPCADataset(Dataset):
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
        data1d = self.data[idx, :] # each row is an expansin !!!
        data_real = data1d.real
        data_imag = data1d.imag
        data_combined = np.concatenate((data_real, data_imag), axis=0)
        # Normalize the data (optional, depending on your model)
        data_combined = (data_combined) / 28  # Scale to [-0.5, 0.5]
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data_combined, dtype=torch.float32)
        # Apply any additional transformations
        if self.transform:
            data_tensor = self.transform(data_tensor)
        label = self.labels[idx]
        label_tensor = torch.tensor(bool(label), dtype=torch.bool)
        return data_tensor, label_tensor

FFPCAdataset = FFPCADataset(expansions, y, Nx, Ny, NBZ)

def PCAtoFF(PCATensordata):
    dataNP = PCATensordata[0].numpy()*28
    coefs = dataNP[0:len(dataNP)//2] + 1j*dataNP[len(dataNP)//2:len(dataNP)]
    FFres = np.zeros((N*N*NBZ),dtype=complex)
    for k in range(len(dataNP)//2):
        FFres += coefs[k]*normalvectors[:,k]
    FFres += barx # very important to add the mean!!!
    return FFres

FFimage1d = PCAtoFF(FFPCAdataset[0])
FFimage2d = utils.convert1Dto2D(FFimage1d,N,NBZ)
BZind = int(NBZ/2)+1 
FFmidtest_phase = np.angle(FFimage2d[:,((BZind-1)*N):(BZind*N)])

plt.figure()
plt.imshow(FFmidtest_phase, cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
#plt.yscale('log')
plt.show()

from torch.utils.data import random_split

writer = None #SummaryWriter(f'runs/mnist/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
# hyperparameters
batch_size = 10
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 100 # 50
latent_dim = 3 #2
hidden_dim = 2048 # 512
train_ratio = 0.8
test_ratio = 1 - train_ratio

dataset_size = len(FFPCAdataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(FFPCAdataset, [train_size, test_size])

# Create a DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
all_loader = DataLoader(FFPCAdataset, shuffle=False)

from src import VAEclass

model = VAEclass.VAE(input_dim=2*Samples,hidden_dim=hidden_dim,latent_dim=latent_dim).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params:,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

prev_updates = 0    
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    prev_updates = VAEclass.train(model,train_loader,optimizer,prev_updates,writer=writer,batch_size=batch_size)
    VAEclass.test(model,test_loader,prev_updates,writer=writer)

# Save the model after training
model_save_path = f'./checkpoints/vae_FFPCA_lat_{latent_dim}_epoch_{num_epochs}.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model
model_load_path = f'./checkpoints/vae_FFPCA_lat_{latent_dim}_epoch_{num_epochs}.pth'
model = VAEclass.VAE(input_dim=2*Samples, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_load_path}")

def PCAdataTophase(Tensordata):
    dataNP = (Tensordata.detach().numpy()-0.5)*28
    coefs = dataNP[0:len(dataNP)//2] + 1j*dataNP[len(dataNP)//2:len(dataNP)]
    FFres = np.zeros((N*N*NBZ),dtype=complex)
    for k in range(len(dataNP)//2):
        FFres += coefs[k]*normalvectors[:,k]
    FFres += barx # very important to add the mean!!!
    FF2d = utils.convert1Dto2D(FFres,N,NBZ)
    # Get the phase of the complex data
    BZind = int(NBZ/2)+1 # choose the center BZ
    data_phase = np.angle(FF2d[:,((BZind-1)*N):(BZind*N)])
    return data_phase

# Get a batch of data from the train_loader
data_iter = iter(all_loader)
images, labels = next(data_iter)

test = model.encode(images[0].to(device))
mean = test.loc  # Mean of the distribution
covariance_matrix = test.covariance_matrix  # Covariance matrix of the distribution
print("Mean:", mean)
print("Covariance Matrix:", covariance_matrix)
imagetest = model.decoder(mean)
plt.figure(figsize=(5, 5))
plt.imshow(PCAdataTophase(imagetest), cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
plt.show()


z = torch.randn(4,latent_dim).to(device)
samples = model.decode(z)
# samples = torch.sigmoid(samples)
# Plot the generated images
fig, ax = plt.subplots(2,2, figsize=(8,8))
for i in range(2):
    for j in range(2):
        img_idx = i*2+j
        im = ax[i,j].imshow(PCAdataTophase(samples[img_idx]),cmap='RdBu', interpolation='nearest')
        #ax[i,j].axis('off')
        #fig.colorbar(im, ax=ax[i, j], orientation='vertical')
plt.show()


# encode and plot the latent space for the training set
model.eval()
z_all = []
y_all = []
with torch.no_grad():
    for data,target in tqdm(all_loader, desc='Encoding'):
        data = data.to(device)
        output = model(data,compute_loss=False)
        z_all.append(output.z_sample.cpu().numpy())
        y_all.append(target.numpy())
z_all = np.concatenate(z_all,axis=0)
y_all = np.concatenate(y_all,axis=0)
# plot classifcation in latent space
plt.figure(figsize=(8, 8))
plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all,cmap='RdBu')
plt.colorbar()
plt.show()
# 3D plot
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(z_all[:, 0], z_all[:, 1], z_all[:, 2], c=y_all, cmap='RdBu')
#plt.colorbar()
#plt.show()


# Interpolating in latent space
n = 10
z1 = torch.linspace(0,1,n)
z2 = torch.linspace(0,2,n)
z = torch.stack([z1,z2],dim=-1).to(device)
samples = model.decode(z)
#samples = torch.sigmoid(samples)
# Plot the generated images
fig, ax = plt.subplots(1,n,figsize=(n,1))
for i in range(n):
    ax[i].imshow(PCAdataTophase(samples[i]),cmap='RdBu')
    ax[i].axis('off')
plt.show()