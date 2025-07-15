from src.VAEclass import VAEOutput
from src.utils import convert1Dto2D

class ConvTransformerVAE(nn.Module):
    """
    Conv-Transformer VAE for form factors without artificial physics constraints.
    The data itself contains the BZ overlap information.
    """
    def __init__(self, input_dim,k_components, hidden_dim, latent_dim,
                 n_heads=8, n_layers=4,d_model=256):
        super(ConvTransformerVAE, self).__init__()
        # input is a 1D form factor
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.k_components = k_components  # Number of k-points
        self.input_dim = input_dim 
        self.matrix_rows = 2*k_components
        self.seq_len = input_dim // (2*k_components)
        self.kernel_size = min(k_components,20)

        # 1D conv along BZ dimension for each k-component
        self.input_conv = nn.Conv1d(self.matrix_rows, d_model, 
                                    kernel_size=self.kernel_size, 
                                    padding=self.kernel_size//2)
        
        # Transformer encoder - decreasing dimensions like FC version
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Encoder progression: decreasing like FC version
        self.encoder_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(hidden_dim),  # Compress sequence to hidden_dim length
            nn.Flatten(),
            nn.Linear(d_model * hidden_dim, hidden_dim),
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
            nn.Linear(hidden_dim, d_model * hidden_dim)
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers//2)
        
        # Output conv to reconstruct form factor
        self.output_conv = nn.Sequential(
            nn.Conv1d(d_model, self.matrix_rows, kernel_size=self.kernel_size, padding=self.kernel_size//2),
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
        x = x.view(batch_size, self.matrix_rows, self.seq_len)

        conv_features = self.input_conv(x)
        # Transpose for transformer: (batch, seq_len, d_model)
        features = conv_features.transpose(1, 2)  # (batch, seq_len, d_model)
        # Transformer encoding
        encoded_features = self.transformer_encoder(features) 
        # Back to conv shape for FC layers
        encoded_features = encoded_features.transpose(1, 2) # (batch, d_model, seq_len)
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
        decoded = self.decoder_fc(z) # (batch, d_model * hidden_dim)
        decoded = decoded.view(batch_size,self.d_model, self.hidden_dim) # (batch, d_model, hidden_dim)
        # Transpose for transformer: (batch, seq_len, d_model)
        decoded_features = decoded.transpose(1,2) # (batch, seq_len, d_model)
        # Create memory for transformer decoder
        memory = decoded_features
        # Create target sequence - should match final sequence length
        target_seq = torch.zeros(batch_size, self.seq_len, self.d_model, device=z.device)
        # Transformer decoding
        decoded_features = self.transformer_decoder(target_seq, memory)
        decoded_features = decoded_features.transpose(1,2) # (batch, d_model, seq_len)
        output = self.output_conv(decoded_features) # (batch, matrix_rows, seq_len)

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