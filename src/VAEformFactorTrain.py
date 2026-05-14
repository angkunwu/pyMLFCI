# source ./.venv/bin/activate
import numpy as np
import torch 
from torch.utils.data import DataLoader, random_split
from src import utils
from src import VAEclass
# import utils
# import VAEclass

#device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

Nx, Ny = 3, 5 #4, 6
train_data, y = utils.ReadAllData(Nx, Ny)  # the train_data
#train_data, y = utils.ReadAllData(Nx, Ny, 
#                                   alphas = np.linspace(0.35,3.55,700),
#                                   c0s1 = np.linspace(-1.0,1.0,100),
#                                   c0s2 = np.linspace(-0.7,0.7,100),
#                                   c0s3 = np.linspace(-0.5,0.5,100))  # the train_data
train_data, y = utils.filterData(train_data,y)
N = Nx * Ny
NBZ = train_data[:,0].shape[0]/N/N
NBZ = int(NBZ)

FFdataset = utils.FormFactorDataset(train_data, y, Nx, Ny, NBZ)
dataset_size = len(FFdataset)
print(f"Dataset size: {dataset_size}")
train_ratio = 1.0
train_size = int(dataset_size * train_ratio)
test_ratio = 1 - train_ratio
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(FFdataset, [train_size, test_size])

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
all_loader = DataLoader(FFdataset, shuffle = False)

# Define the VAE model
hidden_dim = 512 #2048 #32 #2048
latent_dim = 3 #8 #10
print(f"Using hidden_dim: {hidden_dim}, latent_dim: {latent_dim}")
#model = VAEclass.VAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

n_layers = 4
n_heads = 8
#model = VAEclass.TransformerVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, 
#                               latent_dim=latent_dim,n_layers=n_layers,n_heads=n_heads).to(device)
kernel_size = 3
model = VAEclass.BottomConvVAE(input_dim=2*N*N*NBZ, hidden_dim=hidden_dim, 
                               latent_dim=latent_dim, kernel_size=kernel_size).to(device)
"""
d_model = 16
model = VAEclass.ConvTransformerVAE(
    input_dim=2*N*N*NBZ, k_components=N, hidden_dim=hidden_dim, latent_dim=latent_dim, 
    d_model=d_model, n_layers=n_layers,n_heads=n_heads).to(device)
kernel_size = 2
model = VAEclass.SimpleConvVAE(
    input_dim=2*N*N*NBZ, k_components=N, hidden_dim=hidden_dim, 
    latent_dim=latent_dim, kernel_size=kernel_size).to(device)
"""
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in the model: {num_params}")

learning_rate = 1e-3
weight_decay = 1e-2 # regularization
# param = param - lr * (gradient + weight_decay * param)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# not worth it: Reduce LR when loss plateaus
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

num_epochs = 100
prev_updates = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    prev_updates = VAEclass.train(model, train_loader, optimizer, prev_updates, device=device)
    #VAEclass.test(model, test_loader, prev_updates, device=device)

# record final loss
model.eval()
with torch.no_grad():
    total_loss = 0
    total_recon = 0
    total_kl = 0
    for data, target in train_loader:
        data = data.to(device)
        output = model(data)
        total_loss += output.loss.item()
        total_recon += output.loss_recon.item()
        total_kl += output.loss_kl.item()
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)

print(f"Average training loss: {avg_loss}")
print(f"Average reconstruction loss: {avg_recon}")
print(f"Average KL divergence: {avg_kl}")

#model_save_path = f'../checkpoints/vae_FF_lat_{latent_dim}_hid_{hidden_dim}_epoch_{num_epochs}_Nx{Nx}Ny{Ny}.pth'
#model_save_path = f'./checkpoints/vaeMixture_FF_lat_{latent_dim}_hid_{hidden_dim}_epoch_{num_epochs}_Nx{Nx}Ny{Ny}.pth'
#model_save_path = f'./checkpoints/vaeConv_lat_{latent_dim}_hid_{hidden_dim}_kernel_{kernel_size}_Nx{Nx}Ny{Ny}_notest2.pth'
#model_save_path = f'./checkpoints/vaeTrans_lat_{latent_dim}_hid_{hidden_dim}_Nx{Nx}Ny{Ny}.pth'
#torch.save(model.state_dict(), model_save_path)
#print(f"Model saved to {model_save_path}")