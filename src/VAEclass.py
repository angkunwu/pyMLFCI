# use variational autoencoder (VAE) to generate new data and study latent space
# based on https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
#from sklearn.preprocessing import scale
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

def convert1Dto2Dtensor(data1d,N,NBZ):
    """
    Convert a 1D array or tensor to a 2D array or tensor with shape (N, N*NBZ).
    
    Args:
        data1d (numpy.ndarray or torch.Tensor): Input 1D array or tensor of size (N*N*NBZ).
        N (int): Number of components.
        NBZ (int): Number of Brillouin zones.
    
    Returns:
        numpy.ndarray or torch.Tensor: Reshaped 2D array or tensor of shape (N, N*NBZ).
    """
    is_tensor = isinstance(data1d, torch.Tensor)
    # Handle NumPy array
    if not is_tensor:
        data2d = np.zeros((N * NBZ, N), dtype=complex)
        for k in range(NBZ):
            data2d[(k * N):((k + 1) * N), :] = data1d[(k * N * N):((k + 1) * N * N)].reshape(N, N)
        data2d = data2d.T  # Transpose to (N, N*NBZ)
        return data2d
    # Handle PyTorch tensor
    else:
        # Reshape the 1D tensor into (NBZ, N, N)
        data2d = data1d.view(NBZ, N, N)
        # Permute the dimensions to get (N*NBZ, N)
        data2d = data2d.permute(1, 0, 2).reshape(N * NBZ, N)
        # Transpose to get (N, N*NBZ)
        data2d = data2d.T
        return data2d

def reshapeDataBatch(x, N, NBZ):
    """
    Reshape the input data to a 2D format for a batch of data points.
    
    Args:
        x (torch.Tensor): Input data tensor of shape (batch_size, 2 * N * N * NBZ).
        N (int): Number of components.
        NBZ (int): Number of bins.
    
    Returns:
        torch.Tensor: Reshaped data tensor of shape (batch_size, 2 * N, N * NBZ).
    """
    batch_size = x.shape[0]
    # Split the input into real and imaginary parts
    real_part = x[:, :N * N * NBZ].view(batch_size, NBZ, N, N)  # Shape: (batch_size, NBZ, N, N)
    imag_part = x[:, N * N * NBZ:].view(batch_size, NBZ, N, N)  # Shape: (batch_size, NBZ, N, N)
    # Permute and reshape to get (batch_size, N, N * NBZ)
    real_part = real_part.permute(0, 2, 1, 3).reshape(batch_size, N, N * NBZ)
    imag_part = imag_part.permute(0, 2, 1, 3).reshape(batch_size, N, N * NBZ)
    # Concatenate real and imaginary parts along the first dimension
    reshaped_data = torch.cat([real_part, imag_part], dim=1)  # Shape: (batch_size, 2 * N, N * NBZ)

    return reshaped_data

def decodeReshapeData(x, N, NBZ):
    """
    Decode the reshaped data back to the original 1D format.
    
    Args:
        x (torch.Tensor): Input data tensor of shape (batch_size, 2 * N, N * NBZ).
        N (int): Number of components.
        NBZ (int): Number of bins.
    
    Returns:
        torch.Tensor: Decoded data tensor of shape (batch_size, 2 * N * N * NBZ).
    """
    batch_size = x.shape[0]
    # Split the input into real and imaginary parts
    real_part = x[:, :N, :].reshape(batch_size, N, NBZ, N).permute(0, 2, 3, 1)  # Shape: (batch_size, NBZ, N, N)
    imag_part = x[:, N:, :].reshape(batch_size, N, NBZ, N).permute(0, 2, 3, 1)  # Shape: (batch_size, NBZ, N, N)
    # Reshape back to 1D format
    real_part = real_part.reshape(batch_size, -1)  # Shape: (batch_size, N * N * NBZ)
    imag_part = imag_part.reshape(batch_size, -1)  # Shape: (batch_size, N * N * NBZ)
    # Concatenate real and imaginary parts along the last dimension
    decoded_data = torch.cat([real_part, imag_part], dim=1)  # Shape: (batch_size, 2 * N * N * NBZ)

    return decoded_data


@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (Optional[torch.Tensor]): The overall loss of the VAE.
        loss_recon (Optional[torch.Tensor]): The reconstruction loss of the VAE.
        loss_kl (Optional[torch.Tensor]): The KL divergence loss of the VAE.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: Optional[torch.Tensor] = None
    loss_recon: Optional[torch.Tensor] = None
    loss_kl: Optional[torch.Tensor] = None

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
        #scale_tril = torch.diag_embed(scale) # covariance matrix
        #return torch.distributions.MultivariateNormal(mu, scale_tril)
        # Use Independent Normal instead of MultivariateNormal
        base_dist = torch.distributions.Normal(mu, scale)
        return torch.distributions.Independent(base_dist, 1)
    
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
        #std_normal = torch.distributions.MultivariateNormal(
        #    torch.zeros_like(z, device=z.device),
        #    scale_tril = torch.eye(z.shape[-1],device=z.device).unsqueeze(0).expand(z.shape[0],-1,-1),
        #)
        # Use Independent Normal for standard normal prior
        std_normal = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)), 1
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
"""
from dataclasses import dataclass

@dataclass # dataclass decorator
class VAEOutput:
    
    Dataclass for VAE output.

    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss of the VAE.
        loss_kl (torch.Tensor): The KL divergence loss of the VAE.
    
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor
"""


def train(model, dataloader, optimizer, prev_updates, writer=None,device='cpu'):
    """
    Args:
        model(nn.Module): The model to train,
        dataloader (torch.utils.data.DataLoader): The data loader to iterate over the dataset
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters
    """
    #device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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
    print(f'====> Train set loss: {loss.item():.4f}, Recon Loss: {output.loss_recon.item()}, KLD: {output.loss_kl.item():.4f}')
    return prev_updates + len(dataloader)

def test(model, dataloader, cur_step, writer=None,device='cpu'):
    """
    Args:
        model (nn.Module): The model to test
        dataloader (torch.utils.data.DataLoader): The data loader to iterate over the dataset
        cur_step (int): The current step
        writer: The tensorboard writer
    """
    #device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
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






class ConvTransformerVAE(nn.Module):
    """
    Conv-Transformer VAE for form factors without artificial physics constraints.
    The data itself contains the BZ overlap information.
    """
    def __init__(self, input_dim,k_components, hidden_dim, latent_dim,
                 n_heads=8, n_layers=4):
        super(ConvTransformerVAE, self).__init__()
        # input is a 1D form factor
        self.hidden_dim = hidden_dim
        self.k_components = k_components  # Number of k-points
        self.input_dim = input_dim 
        self.matrix_rows = 2*k_components
        self.seq_len = input_dim // (2*k_components)
        self.kernel_size = min(k_components,20)
        self.d_model = self.matrix_rows

        # 1D conv along BZ dimension for each k-component
        self.input_conv = nn.Conv1d(self.seq_len, self.d_model, 
                                    kernel_size=self.kernel_size, 
                                    padding=self.kernel_size//2)
        
        # Transformer encoder - decreasing dimensions like FC version
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Encoder progression: decreasing like FC version
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.d_model * self.matrix_rows, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, 2 * latent_dim)  # 2 for mean and variance
        )
        
        # Decoder: reverse of encoder (increasing dimensions)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.d_model * self.matrix_rows)
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output conv to reconstruct form factor
        self.output_conv = nn.Sequential(
            nn.Conv1d(d_model, self.seq_len, kernel_size=self.kernel_size, padding=self.kernel_size//2),
            nn.Sigmoid()
        )
        
        self.softplus = nn.Softplus()
    
    def encode(self, x, eps: float = 1e-8):
        """
        Encode form factor to latent space.
        Args:
            x (torch.Tensor): Input form factor, shape (batch, 2*15*1905) or (batch, 2, 15, 1905)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, self.matrix_rows, self.seq_len).transpose(1, 2)  # (batch, seq_len, matrix_rows)

        conv_features = self.input_conv(x)
        # Transpose for transformer: (batch, matrix_rows, d_model)
        features = conv_features.transpose(1, 2)  # (batch, matrix_rows, d_model)
        # Transformer encoding
        encoded_features = self.transformer_encoder(features) 
        # Back to conv shape for FC layers
        encoded_features = encoded_features.transpose(1, 2) # (batch, d_model, matrix_rows)
        # FC encoder
        encoded = self.encoder_fc(encoded_features)

        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        scale = self.softplus(logvar) + eps

        base_dist = torch.distributions.Normal(mu, scale)
        return torch.distributions.Independent(base_dist,1)

    def decode(self, z):
        """
        Decode from latent space to form factor.
        Args:
            z (torch.Tensor): Latent variable, shape (batch, latent_dim)
        Returns:
            torch.Tensor: Reconstructed form factor, shape (batch, 2*15*1905)
        """
        batch_size = z.shape[0]
        # FC decoder
        decoded = self.decoder_fc(z) # (batch_size, d_model * matrix_rows)
        decoded = decoded.view(batch_size,self.d_model, self.matrix_rows) # (batch_size, d_model, matrix_rows)
        # Transpose for transformer: (batch, matrix_rows, d_model)
        decoded_features = decoded.transpose(1,2) # (batch, matrix_rows, d_model)
        # Create memory for transformer decoder
        memory = decoded_features
        # Create target sequence - should match final sequence length
        target_seq = torch.zeros(batch_size, self.matrix_rows, self.d_model, device=z.device)
        # Transformer decoding
        decoded_features = self.transformer_decoder(target_seq, memory) # (batch, matrix_rows, d_model)
        decoded_features = decoded_features.transpose(1,2) # (batch, d_model, matrix_rows)
        output = self.output_conv(decoded_features) # (batch, seq_len, matrix_rows)
        output = output.transpose(1,2)  # (batch, matrix_rows, seq_len)
        return output.view(batch_size, -1) # (batch, 2*k_components^2*seq_len)
        

    def reparametrize(self, dist):
        """Reparameterization trick"""
        return dist.rsample()
    
    def forward(self, x, compute_loss: bool = True):
        """Forward pass through the VAE"""
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
        #std_normal = torch.distributions.MultivariateNormal(
        #    torch.zeros_like(z, device=z.device),
        #    scale_tril = torch.eye(z.shape[-1],device=z.device).unsqueeze(0).expand(z.shape[0],-1,-1),
        #)
        # Use Independent Normal for standard normal prior
        std_normal = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)), 1
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
    

class SimpleConvVAE(nn.Module):
    """
    Simplified VAE with Conv1D + Fully Connected layers for form factor learning.
    """
    def __init__(self, input_dim, k_components, hidden_dim, latent_dim, kernel_size=2):
        super(SimpleConvVAE, self).__init__()
        self.k_components = k_components
        self.input_dim = input_dim
        self.matrix_rows = 2 * k_components
        self.seq_len = input_dim // (2 * k_components)
        self.kernel_size = kernel_size
        self.n_filters = self.seq_len #// 2

        # Convolutional Encoder
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(self.seq_len, self.n_filters, kernel_size=self.kernel_size, padding='same'),
            nn.SiLU(),
            nn.Flatten(),
            #nn.Linear(input_dim, self.n_filters * self.matrix_rows),
        )
        # Fully Connected Encoder
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.n_filters * self.matrix_rows, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2 * latent_dim)  # 2 for mean and variance
        )

        # Fully Connected Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_filters * self.matrix_rows),
            nn.SiLU()
        )

        # Convolutional Decoder
        self.conv_decoder = nn.Sequential(
            #nn.Flatten(),
            #nn.Linear(self.n_filters * self.matrix_rows,input_dim),
            #nn.Unflatten(1, (self.seq_len, self.matrix_rows)),  # Reshape to (batch, n_filters, matrix_rows)
            #nn.Unflatten(1, (self.n_filters, self.matrix_rows)),  # Reshape to (batch, n_filters, matrix_rows)
            nn.Conv1d(self.n_filters, self.seq_len, kernel_size=self.kernel_size, padding='same'),
            nn.Sigmoid()
        )

        self.softplus = nn.Softplus()

    def encode(self, x, eps: float = 1e-8):
        """
        Encode input to latent space.
        """
        x = reshapeDataBatch(x, self.k_components, self.seq_len // self.k_components) # (batch, matrix_rows, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, matrix_rows)
        #x = x.view(batch_size, self.matrix_rows, self.seq_len).transpose(1, 2)  # (batch, seq_len, matrix_rows)
        #print(f"Input to Conv1d: {x.shape}") 
        x = self.conv_encoder(x)  # (batch, flattened_dim)
        #print(f"Shape after conv_encoder: {x.shape}")  # Debugging line
        #print(f"supposed input dim: {self.n_filters * self.matrix_rows}")
        x = self.fc_encoder(x)  # (batch, 2 * latent_dim)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        base_dist = torch.distributions.Normal(mu, scale)
        return torch.distributions.Independent(base_dist, 1)

    def decode(self, z):
        """
        Decode latent space to original input space.
        """
        batch_size = z.shape[0]
        x = self.fc_decoder(z) # (batch, n_filters * matrix_rows)
        x = x.view(batch_size, self.matrix_rows, self.n_filters).transpose(1, 2) # (batch, n_filters, matrix_rows)
        x = self.conv_decoder(x)  # (batch, seq_len, matrix_rows)
        x = x.transpose(1, 2)  # (batch, matrix_rows, seq_len)
        x = decodeReshapeData(x, self.k_components, self.seq_len // self.k_components)  # (batch, 2 * k_components^2 * seq_len)
        return x  # (batch, input_dim)

    def reparametrize(self, dist):
        """Reparameterization trick."""
        return dist.rsample()

    def forward(self, x, compute_loss: bool = True):
        """Forward pass through the VAE."""
        dist = self.encode(x)
        z = self.reparametrize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # Compute loss terms
        loss_recon = F.binary_cross_entropy(recon_x, x + 0.5, reduction='none').sum(-1).mean()
        std_normal = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)), 1
        )
        loss_kl = torch.distributions.kl_divergence(dist, std_normal).mean()
        loss = loss_recon + loss_kl
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
    

class BottomConvVAE(nn.Module):
    """
    Bottom convolution VAE with Conv1D + Fully Connected layers for form factor learning.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, kernel_size=2):
        super(BottomConvVAE, self).__init__()

        self.input_dim = input_dim
        self.kernel_size = kernel_size
        dim1 = int(np.sqrt(hidden_dim))
        while hidden_dim % dim1 != 0:
            dim1 -= 1
        dim1p = hidden_dim // dim1
        outpadding = (kernel_size - 1) % 2

        # Fully Connected Encoder + convolutional layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Unflatten(1, (1, dim1, dim1p)), # Reshape to (batch_size, channels=1, height=dim1, width=dim1p)
            nn.Conv2d(1, 4, kernel_size=kernel_size, padding='same'),  # Conv2d layer
            nn.SiLU(),
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(dim1 * dim1p * 4, hidden_dim // 2),  # Adjust dimensions for Conv2d output
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2 * latent_dim)  # 2 for mean and variance
        )
        self.softplus = nn.Softplus()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim1 * dim1p * 4),  # Adjust dimensions for Conv2d output
            nn.SiLU(),
            nn.Unflatten(1, (4, dim1, dim1p)),  # Reshape to (batch_size, channels=4, height=dim1, width=dim1p)
            nn.ConvTranspose2d(4, 1, kernel_size=kernel_size, padding=kernel_size//2, output_padding=outpadding),  # Transpose Conv2d layer
            nn.Flatten(),  # Flatten back to 1D
            nn.Linear(dim1 * dim1p, input_dim),  # Final output layer
            nn.Sigmoid()  # Sigmoid activation for output
        )

    def encode(self, x, eps: float = 1e-8):
        """
        Encode input to latent space.
        """
        x = self.encoder(x)  # (batch, 2 * latent_dim)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        base_dist = torch.distributions.Normal(mu, scale)
        return torch.distributions.Independent(base_dist, 1)

    def decode(self, z):
        """
        Decode latent space to original input space.
        """
        return self.decoder(z)  # (batch, input_dim)
        print(f"Input to decoder: {z.shape}")
        x = self.decoder[0](z)
        print(f"After first Linear layer: {x.shape}")
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        print(f"After second Linear layer: {x.shape}")
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        print(f"After Unflatten: {x.shape}")
        x = self.decoder[5](x)
        print(f"After ConvTranspose2d: {x.shape}")
        x = self.decoder[6](x)
        print(f"After Flatten: {x.shape}")
        x = self.decoder[7](x)
        print(f"After final Linear layer: {x.shape}")
        return x
    
    def reparametrize(self, dist):
        """Reparameterization trick."""
        return dist.rsample()

    def forward(self, x, compute_loss: bool = True):
        """Forward pass through the VAE."""
        dist = self.encode(x)
        z = self.reparametrize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # Compute loss terms
        loss_recon = F.binary_cross_entropy(recon_x, x + 0.5, reduction='none').sum(-1).mean()
        std_normal = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)), 1
        )
        loss_kl = torch.distributions.kl_divergence(dist, std_normal).mean()
        loss = loss_recon + loss_kl
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )