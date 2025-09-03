import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import random
from src import utils
from src import VAEclass
from tqdm.auto import tqdm
import pandas as pd

device = 'cpu'

Nx, Ny = 3,5 
train_data, y = utils.ReadAllData(Nx, Ny)  # the train_data
#train_data, y = utils.filterData(train_data,y)
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

kernel_size = 3
model_load_path = f'./checkpoints/vaeConv_lat_{latent_dim}_hid_{hidden_dim}_kernel_{kernel_size}_Nx{Nx}Ny{Ny}_notest.pth'
model = VAEclass.BottomConvVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim,latent_dim=latent_dim, kernel_size=kernel_size).to(device)

model.load_state_dict(torch.load(model_load_path,map_location=torch.device(device)))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_load_path}")


model.eval()
z_all = []
y_all = []
with torch.no_grad():
    for data,target in tqdm(all_loader, desc='Encoding'):
        data = data.to(device)
        disttemp = model.encode(data) 
        z_all.append(disttemp.base_dist.loc) # get the mean value of each z
        y_all.append(target.numpy())
z_all = np.concatenate(z_all,axis=0)
y_all = np.concatenate(y_all,axis=0)
z_all[49]

maxz = np.max(np.abs(z_all), axis=0)

folder = "~/QuarticCrossing/data/"
pt = folder + str(Nx) + str(Ny) + "data/" 
alphas = np.linspace(0.78943,3.517548,100)
atemp = np.round(alphas[49],5)
#file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(atemp) + ".csv"
file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(atemp) + "Random.csv"
temp = pd.read_csv(file_path)
temp_np = temp.to_numpy()
temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
test_data = temp_np_comp.reshape(-1) # convert into 1d array


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

image = dataToTensor(test_data).unsqueeze(0)
data = image.to(device)
disttemp = model.encode(data) 
meanval = disttemp.base_dist.loc # falls outside the expected range

