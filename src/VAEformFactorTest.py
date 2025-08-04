import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src import VAEclass
from src import utils

device = 'cpu'

Nx, Ny = 3,5 #4,6 #3, 5
train_data, y = utils.ReadAllData(Nx, Ny)  # the train_data
#train_data, y = utils.ReadAllData(Nx, Ny, 
#                                   alphas = np.linspace(0.35,3.55,700),
#                                   c0s1 = np.linspace(-1.0,1.0,100),
#                                   c0s2 = np.linspace(-0.7,0.7,100),
#                                   c0s3 = np.linspace(-0.5,0.5,100))  # the train_data
N = Nx * Ny
NBZ = train_data[:,0].shape[0]/N/N
NBZ = int(NBZ)

FFdataset = utils.FormFactorDataset(train_data, y, Nx, Ny, NBZ)
dataset_size = len(FFdataset)
print(f"Dataset size: {dataset_size}")
all_loader = DataLoader(FFdataset, shuffle = False)

hidden_dim = 2048
latent_dim = 3
num_epochs = 100
# Load the model
#model_load_path = f'./checkpoints/vae_FF_lat_{latent_dim}_hid_{hidden_dim}_decayrate_n2_Nx{Nx}Ny{Ny}.pth'
#model = VAEclass.VAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
kernel_size = 3
model_load_path = f'./checkpoints/vaeConv_lat_{latent_dim}_hid_{hidden_dim}_kernel_{kernel_size}_Nx{Nx}Ny{Ny}.pth'
model = VAEclass.BottomConvVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim,latent_dim=latent_dim, kernel_size=kernel_size).to(device)

#model_load_path = f'./checkpoints/vaeTrans_lat_{latent_dim}_hid_{hidden_dim}_Nx{Nx}Ny{Ny}.pth'
#model = VAEclass.TransformerVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_load_path}")
"""
hidden_dim = 128
latent_dim = 5
d_model = 16
n_layers = 2
n_heads = 4
model_save_path = f'./checkpoints/vaeMixture_FF_lat_{latent_dim}_hid_{hidden_dim}_epoch_{num_epochs}.pth'
model = VAEclass.ConvTransformerVAE(
    input_dim=2*N*N*NBZ, k_components=N, hidden_dim=hidden_dim, latent_dim=latent_dim, 
    d_model=d_model, n_layers=n_layers,n_heads=n_heads).to(device)
model.load_state_dict(torch.load(model_load_path))
"""

def dataTophase(Tensordata):
    # Convert the data to 2D with complex numbers
    dataNP = (Tensordata.detach().numpy()-0.5)*2.0 # -0.5 from adding 0.5 in the model class
    dataComplex = utils.convert1Dto2D(dataNP[:(N*N*NBZ)],N,NBZ) + 1j*utils.convert1Dto2D(dataNP[(N*N*NBZ):],N,NBZ)
    # Get the phase of the complex data
    BZind = int(NBZ/2)+1 # choose the center BZ
    data_phase = np.angle(dataComplex[:,((BZind-1)*N):(BZind*N)])
    return data_phase

def dataToTensor(npdata,shift=False):
    data_real = npdata.real
    data_imag = npdata.imag
    data_combined = np.concatenate((data_real, data_imag), axis = 0)
    # Normalize the data (optional, depending on your model)
    data_combined = (data_combined) / 2.0  # Scale to [-0.5, 0.5]
    if shift:
        data_combined += 0.5  # Shift to [0, 1] range
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data_combined, dtype=torch.float32)
    return data_tensor

FormFactor2d = utils.convert1Dto2D(train_data[:,0],N,NBZ)
BZind = int(NBZ/2)+1 # choose the center BZ
FFmiddle = FormFactor2d[:,((BZind-1)*N):(BZind*N)]
FFmiddle_phase = np.angle(FFmiddle)

plt.figure()
plt.imshow(FFmiddle_phase, cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
#plt.yscale('log')
plt.show()

# Get a batch of data from the train_loader
data_iter = iter(all_loader)
images, labels = next(data_iter)

test = model.encode(images[0].unsqueeze(0).to(device))  # Encode the first image in the batch
#mean = test.loc  # Mean of the distribution
#covariance_matrix = test.covariance_matrix  # Covariance matrix of the distribution
mean = test.base_dist.loc
variance = test.base_dist.scale ** 2 
print("Mean:", mean)
#print("Covariance Matrix:", covariance_matrix)
print("Variance:", variance)
imagetest = model.decoder(mean).squeeze(0).cpu()  # Decode the mean to get the reconstructed image
plt.figure(figsize=(5, 5))
plt.imshow(dataTophase(imagetest), cmap='RdBu', interpolation='nearest')
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
        ax[i,j].imshow(dataTophase(samples[img_idx]),cmap='RdBu', interpolation='nearest')
        ax[i,j].axis('off')
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

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(z_all[:, 0], z_all[:, 1], z_all[:, 2], c=y_all, cmap='RdBu')
# Add color bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('IsFCI')
ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_zlabel('Latent Dimension 3')
plt.show()

# Interpolating in latent space
n = 8
z1 = torch.linspace(1.46,-0.76,n)
z2 = torch.linspace(0.79,-1.33,n)
z3 = torch.linspace(0.46,0.74,n)
z = torch.stack([z1,z2,z3],dim=-1).to(device)
samples = model.decode(z)
#samples = torch.sigmoid(samples)
# Plot the generated images
fig, ax = plt.subplots(1,n,figsize=(n,1))
for i in range(n):
    ax[i].imshow(dataTophase(samples[i]),cmap='RdBu')
    ax[i].axis('off')
plt.show()

alphas = np.linspace(0.78943,3.517548,100) # ind=0,49,99
z0 = torch.tensor(z_all[0], dtype=torch.float32).to(device)
z1 = torch.tensor(z_all[99], dtype=torch.float32).to(device)
# generate a linear interpolation between z0 and z1
n = 20
alpha = torch.linspace(0, 1, n).unsqueeze(1).to(device)  # Shape: (n, 1)
z_interp = (1 - alpha) * z0.unsqueeze(0) + alpha * z1.unsqueeze(0)  # Broadcasting
samples_interp = model.decode(z_interp)
fig, ax = plt.subplots(1,n,figsize=(n,1))
for i in range(n):
    ax[i].imshow(dataTophase(samples_interp[i]),cmap='RdBu')
    ax[i].axis('off')
plt.show()

#alphas = np.linspace(0.78943,3.517548,100) # ind=0,49,99
model.eval()
image = dataToTensor(train_data[:,49]).unsqueeze(0)
zs = np.zeros((n, latent_dim))
with torch.no_grad():
    for i in range(n):
        output = model(image, compute_loss=False)
        zs[i] = output.z_sample.cpu().numpy()

samples_interp = model.decode(torch.tensor(zs, dtype=torch.float32).to(device))


samples_interp = model.decode(torch.tensor(z_all[:100], dtype=torch.float32).to(device))
samples_interp = torch.zeros_like(samples_interp)
for i in range(100):
    samples_interp[i] = dataToTensor(train_data[:,i],shift=True)

maxz = np.max(np.abs(z_all), axis=0)
num_samples = 100
# Generate random samples in the range [-maxz[i], maxz[i]] for each dimension
z_new = torch.zeros((num_samples, latent_dim), dtype=torch.float32)
for i in range(latent_dim):
    z_new[:, i] = torch.FloatTensor(num_samples).uniform_(-maxz[i]*1.1, maxz[i]*1.1)
samples_interp = model.decode(z_new)

import pandas as pd

def outputFF(samples, idx):
    dataNP = (samples[idx].detach().numpy()-0.5)*2.0 # -0.5 from adding 0.5 in the model class
    FFfull = utils.convert1Dto2D(dataNP[:(N*N*NBZ)],N,NBZ) + 1j*utils.convert1Dto2D(dataNP[(N*N*NBZ):],N,NBZ)
    
    FFfull_real = np.real(FFfull).transpose()
    FFfull_imag = np.imag(FFfull).transpose()

    headerRe = ["K"+str(k)+"Re" for k in range(1,N+1)]
    headerIm = ["K"+str(k)+"Im" for k in range(1,N+1)]
    header = headerRe + headerIm
    outMat = np.zeros((FFfull_real.shape[0],2*N))
    outMat[:,0:N] = FFfull_real
    outMat[:,N:2*N] = FFfull_imag
    df = pd.DataFrame(outMat,columns=header)

    # save the df to a .csv file
    pt = "~/pyMLFCI/data/"
    file_path = pt + "FFVAESample" + str(idx) + ".csv"
    df.to_csv(file_path,index=False)
    return

for k in range(100): 
    outputFF(samples_interp, k)


# compare training data and generated data
realdata = utils.convert1Dto2D(train_data[:,0],N,NBZ)
dataNP = (samples_interp[0].detach().numpy()-0.5)*2.0 # -0.5 from adding 0.5 in the model class
gendata = utils.convert1Dto2D(dataNP[:(N*N*NBZ)],N,NBZ) + 1j*utils.convert1Dto2D(dataNP[(N*N*NBZ):],N,NBZ)

np.abs(gendata - realdata)
# find the maximum absolute difference
max_diff = np.max(np.abs(gendata - realdata))
max_diff
