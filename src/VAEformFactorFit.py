import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src import VAEclass
from src import utils
from src import FormFactorFuns as FFFs
import pandas as pd

def TensorTo2Ddata(Tensordata):
    # Convert the data to 2D with complex numbers
    dataNP = (Tensordata.detach().numpy()-0.5)*2.0 # -0.5 from adding 0.5 in the model class
    dataComplex = utils.convert1Dto2D(dataNP[:(N*N*NBZ)],N,NBZ) + 1j*utils.convert1Dto2D(dataNP[(N*N*NBZ):],N,NBZ)
    return dataComplex

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

device = 'cpu'

Nx, Ny = 4,6 #3, 5
#train_data, y = utils.ReadAllData(Nx, Ny)  # the train_data
train_data, y = utils.ReadAllData(Nx, Ny, 
                                    alphas = np.linspace(0.35,3.55,700),
                                    c0s1 = np.linspace(-1.0,1.0,100),
                                    c0s2 = np.linspace(-0.7,0.7,100),
                                    c0s3 = np.linspace(-0.5,0.5,100))  # the train_data
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
model_load_path = f'./checkpoints/vaeConv_lat_{latent_dim}_hid_{hidden_dim}_kernel_{kernel_size}_Nx{Nx}Ny{Ny}all_notest.pth'
model = VAEclass.BottomConvVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim,latent_dim=latent_dim, kernel_size=kernel_size).to(device)

model.load_state_dict(torch.load(model_load_path,map_location=torch.device(device)))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded from {model_load_path}")

# find the magic angle z values in the latent space
model.eval()  # Set the model to evaluation mode
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

z0 = torch.tensor(z_all[96], dtype=torch.float32).to(device) #[ 0.4358, -0.2916, -1.4175]
z1 = torch.tensor(z_all[389], dtype=torch.float32).to(device) #[-0.3054,  1.0327,  0.6870]
z2 = torch.tensor(z_all[692], dtype=torch.float32).to(device) #[-0.5341, -0.4416,  1.7357]

# sample all range of z values from the tranining data, obtain maxz
model.eval()  # Set the model to evaluation mode
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

maxz = np.max(np.abs(z_all), axis=0)

# target form factor
Ind = 23
pt = "~/pyMLFCI/data/"
file_path = pt + "FFVAESampleNx"+ str(Nx) + "Ny" + str(Ny) + "Ind"+ str(Ind) +"All.csv"
temp = pd.read_csv(file_path)
temp_np = temp.to_numpy()
temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
test_data = temp_np_comp.reshape(-1) # convert into 1d array
test_2d = utils.convert1Dto2D(test_data,N,NBZ)
# Bcurs = FFFs.WilsonLoopFull(Nx, Ny, test_2d, etaxy=False)
# trg1s = FFFs.WilsonLoopFull(Nx, Ny, test_2d, etaxy=True)
# Ctemp = sum(Bcurs)/2/np.pi
# trgtemp = sum(trg1s)/2/np.pi/2



def progressive_latent_search(model, target_2d,maxz, 
                              n_iter=10, n_samples=1000, verbose=True):
    """
    Progressive random search in latent space to match target_2d.
    Args:
        model: Trained VAE model with .decode(z)
        target_2d: Target form factor (2D numpy array, complex)
        N, NBZ: Lattice/BZ sizes
        maxz: Initial search range (array-like, shape [latent_dim])
        n_iter: Number of progressive search iterations
        n_samples: Number of samples per iteration
        verbose: Print progress
    Returns:
        best_z: Found latent vector (torch tensor)
        best_loss: Final loss value
    """
    # Start at center
    center = torch.zeros(latent_dim, device=device)
    box = torch.tensor(maxz, dtype=torch.float32, device=device)
    best_z = None
    best_loss = float('inf')
    for it in range(n_iter):
        zs = torch.empty((n_samples, latent_dim), device=device)
        for i in range(latent_dim):
            zs[:, i] = torch.empty(n_samples).uniform_(
                (center[i] - box[i]).item(), (center[i] + box[i]).item()
            )
        losses = []
        for j in tqdm(range(n_samples)):
            z = zs[j].unsqueeze(0)
            with torch.no_grad():
                decoded = model.decode(z)
            # Convert to 2D complex
            decoded2d = TensorTo2Ddata(decoded[0])
            loss = (abs(target_2d - decoded2d)**2).sum()
            losses.append(loss)
        losses = np.array(losses)
        min_idx = np.argmin(losses)
        if losses[min_idx] < best_loss:
            best_loss = losses[min_idx]
            best_z = zs[min_idx].clone().detach()
        # Update center and box for next iteration
        center = zs[min_idx]
        box = box / 2.0
        if verbose:
            print(f"Iter {it+1}/{n_iter}: Best loss {best_loss:.6g}, z={center.cpu().numpy()}, box={box.cpu().numpy()}")
    return best_z, best_loss


best_z,best_loss = progressive_latent_search(model, test_2d, maxz, n_iter=10, n_samples=1000)
# [-6.0038,  0.9378, -0.0133]


# image = dataToTensor(test_data).unsqueeze(0)
# data = image.to(device)
# with torch.no_grad():
#      disttemp = model.encode(data) 
#      meanval = disttemp.base_dist.loc # falls outside the expected range
# # take a sum of abs(meanval-z0)**2
# (abs(meanval - z1)**2).detach().numpy().sum()


imagetemp = model.decode(best_z.unsqueeze(0))
Component2d = TensorTo2Ddata(imagetemp[0])
Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
trg1s = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
Ctemp = sum(Bcurs)/2/np.pi
trgtemp = sum(trg1s)/2/np.pi/2

(abs(test_2d - Component2d)**2).sum()

def npdataTophase(npdata):
    dataComplex = utils.convert1Dto2D(npdata,N,NBZ)
    # Get the phase of the complex data
    BZind = int(NBZ/2)+1 # choose the center BZ
    data_phase = np.angle(dataComplex[:,((BZind-1)*N):(BZind*N)])
    return data_phase

def dataTophase(Tensordata):
    # Convert the data to 2D with complex numbers
    dataNP = (Tensordata.detach().numpy()-0.5)*2.0 # -0.5 from adding 0.5 in the model class
    dataComplex = utils.convert1Dto2D(dataNP[:(N*N*NBZ)],N,NBZ) + 1j*utils.convert1Dto2D(dataNP[(N*N*NBZ):],N,NBZ)
    # Get the phase of the complex data
    BZind = int(NBZ/2)+1 # choose the center BZ
    data_phase = np.angle(dataComplex[:,((BZind-1)*N):(BZind*N)])
    return data_phase

fig, ax = plt.subplots(1,2,figsize=(10,1))
ax[0].imshow(npdataTophase(test_data),cmap='RdBu')
ax[1].imshow(dataTophase(imagetemp[0]),cmap='RdBu')
plt.show()


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
    #file_path = pt + "FFVAESample" + str(idx) + ".csv"
    #file_path = pt + "FFVAESampleNx4Ny6Ind" + str(idx) + "Interpolate.csv"
    file_path = pt + "FFVAESampleNx4Ny6Ind" + str(idx) + "FCICDW.csv"
    df.to_csv(file_path,index=False)
    return


(abs(best_z - z1)**2).detach().numpy().sum()

n = 100
alpha = torch.linspace(0, 1, n).unsqueeze(1).to(device)  # Shape: (n, 1)
z_interp = (1 - alpha) * z1.unsqueeze(0) + alpha * best_z.unsqueeze(0)  # Broadcasting
#z_interp = z_interp.unsqueeze(0)  
samples_interp = model.decode(z_interp)

for k in range(100):
    outputFF(samples_interp, k)


