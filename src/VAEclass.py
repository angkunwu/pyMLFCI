# use variational autoencoder (VAE) to generate new data and study latent space
# based on https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.
    Args:
        input_dim (int): dimension of input data
        hidden_dim (int): dimension of hidden layers
        latent_dim (int): dimension of latent space
    """
    def __init__(self,input_dim, hidden_dim,latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_dim,hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, 2*latent_dim), # 2 for mean and variance.
        )

        self.softplus = nn.Softplus() # for variance
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input into the latent space.
        Args:
            x (torch.Tensor): Input data
            eps (float): Small value to avoid division by zero
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1) # mean and log variance
        scale = self.softplus(logvar) + eps # standard deviation
        scale_tril = torch.diag_embed(scale) # covariance matrix
        return torch.distributions.MultivariateNormal(mu, scale_tril)
    
    def reparametrize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data
        Returns:
            torch.Tensor: Sampled data from the latent space
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent sapce to the original input space.
        Args:
            z (torch.Tensor): Data in the latent space
        Returns:
            torch.Tensor: Reconstructed data in the original input space
        """
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True):
        """
        Perform a forward pass of the VAE
        Args:
            x (torch.Tensor): Input data
            compute_loss (bool): Whether to compute the loss
        Returns:
            VAEOutput: VAE output dataclass
        """
        dist = self.encode(x)
        z = self.reparametrize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist = dist,
                z_sample = z,
                x_recon = recon_x,
                loss = None,
                loss_recon = None,
                loss_kl = None,
            )
        # compute loss terms
        loss_recon = F.binary_cross_entropy(recon_x, x+0.5, reduction='none').sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril = torch.eye(z.shape[-1],device=z.device).unsqueeze(0).expand(z.shape[0],-1,-1),
        )
        loss_kl = torch.distributions.kl_divergence(dist, std_normal).mean()
        loss = loss_recon + loss_kl
        return VAEOutput(
            z_dist = dist,
            z_sample = z,
            x_recon = recon_x,
            loss = loss,
            loss_recon = loss_recon,
            loss_kl = loss_kl,
        )

from dataclasses import dataclass

@dataclass # dataclass decorator
class VAEOutput:
    """
    Dataclass for VAE output.

    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss of the VAE.
        loss_kl (torch.Tensor): The KL divergence loss of the VAE.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor



def train(model, dataloader, optimizer, prev_updates, writer=None,batch_size=128):
    """
    Args:
        model(nn.Module): The model to train,
        dataloader (torch.utils.data.DataLoader): The data loader to iterate over the dataset
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters
    """
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    model.train() # Set the model to training mode
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx # number of updates
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data) # forward pass
        loss = output.loss # calculate the loss
        loss.backward()
        if n_upd % 100 == 0:
            # Calculate and log the gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2) # L2 norm
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            #print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')
            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/train', total_norm, global_step)
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return prev_updates + len(dataloader)

def test(model, dataloader, cur_step, writer=None):
    """
    Args:
        model (nn.Module): The model to test
        dataloader (torch.utils.data.DataLoader): The data loader to iterate over the dataset
        cur_step (int): The current step
        writer: The tensorboard writer
    """
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    model.eval() # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    with torch.no_grad(): # Disable gradient computation to save memory
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            data = data.view(data.size(0),-1) # Flatten the data
            output = model(data,compute_loss=True) # forward pass
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
    
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss}, KLD: {test_kl_loss})')
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/test/KLD', output.loss_kl.item(), global_step=cur_step)
        # Log reconstructions
        writer.add_images('Test/Reconstructions',output.x_recon.view(-1,1,28,28),global_step=cur_step)
        writer.add_images('Test/Originals',data.view(-1,1,28,28),global_step=cur_step)
        # Log random samples from the latent space
        z = torch.randn(16,latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples',samples.view(-1,1,28,28),global_step=cur_step)
