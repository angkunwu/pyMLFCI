import torch
from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '~/.pytorch/MNIST_data/', #?????
    train_batch_size = 32,
    train_lr = 1e-3,
    train_num_steps = 100,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)