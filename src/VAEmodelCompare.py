import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import random
from src import utils
from src import VAEclass

device = 'cpu'

Nx, Ny = 3,5 #3, 5 #4, 6
train_data, y = utils.ReadAllData(Nx, Ny)  # the train_data
# train_data, y = utils.ReadAllData(Nx, Ny, 
#                                    alphas = np.linspace(0.35,3.55,700),
#                                    c0s1 = np.linspace(-1.0,1.0,100),
#                                    c0s2 = np.linspace(-0.7,0.7,100),
#                                    c0s3 = np.linspace(-0.5,0.5,100))  # the train_data
train_data, y = utils.filterData(train_data,y)
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
model_load_path = f'./checkpoints/vae_FF_lat_{latent_dim}_hid_{hidden_dim}_decayrate_n2_Nx{Nx}Ny{Ny}_notest.pth'
model = VAEclass.VAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
kernel_size = 3
model_load_path = f'./checkpoints/vaeConv_lat_{latent_dim}_hid_{hidden_dim}_kernel_{kernel_size}_Nx{Nx}Ny{Ny}_notest.pth'
model = VAEclass.BottomConvVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim,latent_dim=latent_dim, kernel_size=kernel_size).to(device)

#model_load_path = f'./checkpoints/vaeTrans_lat_{latent_dim}_hid_{hidden_dim}_Nx{Nx}Ny{Ny}.pth'
#model = VAEclass.TransformerVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

model.load_state_dict(torch.load(model_load_path,map_location=torch.device(device)))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_load_path}")

# record final loss
model.eval()
with torch.no_grad():
    total_loss = 0
    total_recon = 0
    total_kl = 0
    for data, target in all_loader:
        data = data.to(device)
        output = model(data)
        total_loss += output.loss.item()
        total_recon += output.loss_recon.item()
        total_kl += output.loss_kl.item()
    avg_loss = total_loss / len(all_loader)
    avg_recon = total_recon / len(all_loader)
    avg_kl = total_kl / len(all_loader)

print(f"Average training loss: {avg_loss}")
print(f"Average reconstruction loss: {avg_recon}")
print(f"Average KL divergence: {avg_kl}")

