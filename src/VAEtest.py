import torch
from torchvision import datasets
from torchvision.transforms import v2

batch_size = 128
tranform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x.view(-1) - 0.5),
])

# Download the MNIST dataset
train_data = datasets.MNIST(
    '~/.pytorch/MNIST_data/',
    download=True,
    train=True,
    transform=tranform,
)
# test data
test_data = datasets.MNIST(
    '~/.pytorch/MNIST_data/',
    download=True,
    train=False,
    transform=tranform,
)
# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True,
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = batch_size,
    shuffle = False,
)

# visualize some data
import matplotlib.pyplot as plt
import numpy as np

# Get a batch of data from the train_loader
data_iter = iter(train_loader)
images, labels = next(data_iter)
# Plot the first 64 images in the batch
fig, ax = plt.subplots(8, 8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        img_idx = i * 8 + j
        ax[i, j].imshow(images[img_idx].view(28, 28).cpu().numpy(), cmap='gray')
        ax[i, j].axis('off')

plt.show()


#from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

writer = None #SummaryWriter(f'runs/mnist/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hyperparameters
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 10 # 50
latent_dim = 2
hidden_dim = 512

from src import VAEclass

model = VAEclass.VAE(input_dim=784,hidden_dim=hidden_dim,latent_dim=latent_dim).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params:,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

prev_updates = 0    
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    prev_updates = VAEclass.train(model,train_loader,optimizer,prev_updates,writer=writer,batch_size=batch_size)
    VAEclass.test(model,test_loader,prev_updates,writer=writer)

z = torch.randn(64,latent_dim).to(device)
samples = model.decode(z)
# samples = torch.sigmoid(samples)
# Plot the generated images
fig, ax = plt.subplots(8,8, figsize=(8,8))
for i in range(8):
    for j in range(8):
        img_idx = i*8+j
        ax[i,j].imshow(samples[img_idx].view(28,28).cpu().detach().numpy(),cmap='gray')
        ax[i,j].axis('off')
plt.show()

# encode and plot the latent space for the training set
model.eval()
z_all = []
y_all = []
with torch.no_grad():
    for data,target in tqdm(train_loader, desc='Encoding'):
        data = data.to(device)
        output = model(data,compute_loss=False)
        z_all.append(output.z_sample.cpu().numpy())
        y_all.append(target.numpy())
z_all = np.concatenate(z_all,axis=0)
y_all = np.concatenate(y_all,axis=0)
# plot classifcation in latent space
plt.figure(figsize=(10, 10))
plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='tab10')
plt.colorbar()
plt.show()

# Interpolating in latent space
n = 15
z1 = torch.linspace(-0,1,n)
z2 = torch.zeros_like(z1)+2
z = torch.stack([z1,z2],dim=-1).to(device)
samples = model.decode(z)
samples = torch.sigmoid(samples)
# Plot the generated images
fig, ax = plt.subplots(1,n,figsize=(n,1))
for i in range(n):
    ax[i].imshow(samples[i].view(28,28).cpu().detach().numpy(),cmap='gray')
    ax[i].axis('off')
plt.show()
