# source ./.venv/bin/activate
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import utils
import VAEclass

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
Nx, Ny = 3, 5
train_data, y = utils.ReadAllData(Nx, Ny)
N = Nx * Ny
NBZ = train_data[:, 0].shape[0] // (N * N)

FFdataset = utils.FormFactorDataset(train_data, y, Nx, Ny, NBZ)
dataset_size = len(FFdataset)
print(f"Dataset size: {dataset_size}")
train_ratio = 0.9
train_size = int(dataset_size * train_ratio)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(FFdataset, [train_size, test_size])

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
hidden_dim = 128 #2048
#latent_dims = [5,10, 15, 20, 25, 30, 40, 50] # [2,3,4,5,10,20,30,40,50]
latent_dims = [2,3,4,5,10,20,30,40]
learning_rate = 1e-3
weight_decay = 1e-3
num_epochs = 100

d_model = 16
n_layers = 2
n_heads = 4

# Iterate over latent_dim values
for latent_dim in latent_dims:
    print(f"Training model with latent_dim: {latent_dim}")
    
    # Define the VAE model
    #model = VAEclass.VAE(input_dim=2 * N * N * NBZ, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model = VAEclass.ConvTransformerVAE(input_dim=2*N*N*NBZ, k_components=N, hidden_dim=hidden_dim, 
                                        latent_dim=latent_dim, d_model=d_model, n_layers=n_layers,
                                        n_heads=n_heads).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {num_params}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    prev_updates = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} for latent_dim {latent_dim}")
        prev_updates = VAEclass.train(model, train_loader, optimizer, prev_updates, device=device)
        VAEclass.test(model, test_loader, prev_updates, device=device)
    
    # Save the model
    #model_save_path = f'../checkpoints/vae_FF_lat_{latent_dim}_hid_{hidden_dim}_epoch_{num_epochs}.pth'
    model_save_path = f'../checkpoints/vaeMixture_lat_{latent_dim}_hid_{hidden_dim}_dmodel_{d_model}_nhead_{n_heads}_nlayer_{n_layers}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model with latent_dim {latent_dim} saved to {model_save_path}")