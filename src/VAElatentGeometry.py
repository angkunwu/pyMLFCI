import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src import VAEclass
from src import utils
from src import FormFactorFuns as FFFs

device = 'cpu'

Nx, Ny = 4,6
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

hidden_dim = 2048 #1024 #2048
latent_dim = 3
num_epochs = 100

kernel_size = 3 #3
model_load_path = f'./checkpoints/vaeConv_lat_{latent_dim}_hid_{hidden_dim}_kernel_{kernel_size}_Nx{Nx}Ny{Ny}_notest.pth'
model = VAEclass.BottomConvVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim,latent_dim=latent_dim, kernel_size=kernel_size).to(device)

model.load_state_dict(torch.load(model_load_path,map_location=torch.device(device)))
print(f"Model loaded from {model_load_path}")

def TensorTo2Ddata(Tensordata):
    # Convert the data to 2D with complex numbers
    dataNP = (Tensordata.detach().numpy()-0.5)*2.0 # -0.5 from adding 0.5 in the model class
    dataComplex = utils.convert1Dto2D(dataNP[:(N*N*NBZ)],N,NBZ) + 1j*utils.convert1Dto2D(dataNP[(N*N*NBZ):],N,NBZ)
    return dataComplex


model.eval()  # Set the model to evaluation mode
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

maxz = np.max(np.abs(z_all), axis=0)
Samples = 100
# Generate random samples in the range [-maxz[i], maxz[i]] for each dimension
z_new = torch.zeros((Samples, latent_dim), dtype=torch.float32)
for i in range(latent_dim):
    z_new[:, i] = torch.FloatTensor(Samples).uniform_(-maxz[i]*1.1, maxz[i]*1.1)
samples_interp = model.decode(z_new)



Fxys = np.zeros(Samples)
trg1 = np.zeros(Samples)
Fxystd = np.zeros(Samples)
trg1std = np.zeros(Samples)
for k in range(Samples):
    Component2d = TensorTo2Ddata(samples_interp[k])
    Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
    trg1s = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
    Fxys[k] = sum(Bcurs)/2/np.pi
    trg1[k] = sum(trg1s)/2/np.pi/2
    Fxystd[k] = np.std(Bcurs)/2/np.pi
    trg1std[k] = np.std(trg1s)/2/np.pi/2


plt.figure()
plt.scatter(np.arange(1,Samples+1),Fxys,label=r'$C_B$',s=10)
plt.scatter(np.arange(1,Samples+1),trg1,label=r'$C_{tr(g)}$',s=5)
plt.scatter(np.arange(1,Samples+1),Fxystd*8,label=r'$8\sigma_{C_B}$',s=10)
plt.scatter(np.arange(1,Samples+1),trg1std*8,label=r'$8\sigma_{C_{tr(g)}}$',s=10)
plt.xlabel(r'sample $j$',fontsize=12)
plt.ylabel(r'geometric quantities',fontsize=12)
plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 0.95))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-1,50)
plt.show()
# save to a .pdf file to the Desktop
#pt = "/Users/angkunwu/Desktop/"
#plt.savefig(pt+'expansion.pdf',format='pdf')

sortinds = np.argsort(trg1)

plt.figure()
plt.scatter(np.arange(1,Samples+1),-Fxys[sortinds],label=r'$C_B$',s=10)
plt.scatter(np.arange(1,Samples+1),trg1[sortinds],label=r'$C_{tr(g)}$',s=5)
plt.scatter(np.arange(1,Samples+1),Fxystd[sortinds]*8,label=r'$8\sigma_{C_B}$',s=5)
plt.scatter(np.arange(1,Samples+1),trg1std[sortinds]*8,label=r'$8\sigma_{C_{tr(g)}}$',s=3) 
plt.xlabel(r'sample $j$',fontsize=12)
plt.ylabel(r'geometric quantities',fontsize=12)
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.1, 0.95))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
#sc = ax.scatter(z_new[sortinds, 0], z_new[sortinds, 1], z_new[sortinds, 2],c=trg1[sortinds], cmap='RdBu')
sc = ax.scatter(z_all[:, 0], z_all[:, 1], z_all[:, 2],c=y_all, cmap='RdBu')
# Add color bar
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label(r'$C_{tr(g)}$')
ax.set_xlabel(r'$z_1$')
ax.set_ylabel(r'$z_2$')
ax.set_zlabel(r'$z_3$')
plt.show()

# # plot histogram of trg1std
# plt.figure()
# plt.hist(trg1std, bins=20, alpha=0.7, label=r'$C_{tr(g)}$')
# plt.xlabel(r'$C_{tr(g)}$')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()

Samples = 100
z_new = torch.zeros((Samples, latent_dim), dtype=torch.float32)
Fxys = np.zeros(Samples)
trg1 = np.zeros(Samples)
Fxystd = np.zeros(Samples)
trg1std = np.zeros(Samples)
cursam = 0
runind = 0
while cursam < Samples:
    z_temp = torch.zeros((1, latent_dim), dtype=torch.float32)
    for i in range(latent_dim):
        #z_temp[:,i] = torch.FloatTensor(1).uniform_(-maxz[i]*1.1, maxz[i]*1.1)
        z_temp[:,i] = torch.FloatTensor(1).uniform_(-maxz[i]*1.0, maxz[i]*1.0)
    imagetemp = model.decode(z_temp)
    Component2d = TensorTo2Ddata(imagetemp[0])
    Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
    trg1s = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
    Ctemp = sum(Bcurs)/2/np.pi
    trgtemp = sum(trg1s)/2/np.pi/2
    stdtrg = np.std(trg1s)/2/np.pi/2
    if trgtemp > 1.05 and np.abs(Ctemp+1) < 0.01 and stdtrg < 0.015: #stdtrg < 0.005:
    #if np.abs(Ctemp+1) > 0.5:
    #if trgtemp > 2.9 and trgtemp < 3.1 and np.abs(Ctemp+1) < 0.01:
        z_new[cursam] = z_temp
        Fxys[cursam] = Ctemp
        trg1[cursam] = trgtemp
        Fxystd[cursam] = np.std(Bcurs)/2/np.pi
        trg1std[cursam] = np.std(trg1s)/2/np.pi/2
        cursam += 1
        print(Ctemp)
    runind += 1
    print(runind, ' ',cursam)



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
    #file_path = pt + "FFVAESampleNx3Ny5Ind" + str(idx) + ".csv"
    #file_path = pt + "FFVAESampleNx4Ny6Ind" + str(idx) + "all.csv"
    file_path = pt + "FFVAESampleNx4Ny6Ind" + str(idx) + "Lowg.csv"
    df.to_csv(file_path,index=False)
    return

samples_interp = model.decode(torch.tensor(z_new, dtype=torch.float32).to(device))

for k in range(100): 
    outputFF(samples_interp, k)


# # remove files
# import os
# pt = os.path.expanduser("~/pyMLFCI/data/")
# for k in range(100): 
#     file_path = pt + "FFVAESampleNx4Ny6" + str(k) + ".csv"
#     if os.path.exists(file_path):
#         os.remove(file_path)
#     else:
#         print(f"The file {file_path} does not exist.")