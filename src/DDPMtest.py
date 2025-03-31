# Add the current directory  to the Python path
import sys,os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torch.optim as optim
from src.DDPMclass import UNET,DDPM_Scheduler, set_seed
from src.DDPMclass import train, display_reverse
from timm.utils import ModelEmaV3
import numpy as np
import random
import math
import pdb
from tqdm import tqdm
from typing import List

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def inference(checkpoint_path:str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999,
              device: torch.device='cpu',
              ):

    checkpoint = torch.load(checkpoint_path,map_location=device) # Load the checkpoint from the specified path
    model = UNET().to(device)
    model.load_state_dict(checkpoint['weights']) # Load the model weights from the checkpoint
    ema = ModelEmaV3(model, decay=ema_decay) # Initialize the EMA model
    ema.load_state_dict(checkpoint['ema']) # Load the EMA state
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps) # Initialize the scheduler
    times = [0, 15,50,100,200,300,400,550,700,999]
    images = []
    with torch.no_grad():# Disable gradient computation for inference
        model = ema.module.eval()
        z = torch.randn(1,1,32,32).to(device)
        for t in reversed(range(1,num_time_steps)):
            t = [t]
            temp = (scheduler.beta[t]/((torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
            z = (1/(torch.sqrt(1-scheduler.beta[t]))).to(device)*z-(temp*model(z,t))
            if t[0] in times:
                images.append(z.cpu())
            e = torch.randn(1,1,32,32).to(device)
            z = z + (e*torch.sqrt(scheduler.beta[t]).to(device))
        temp = scheduler.beta[0]/((torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0]))).to(device)
        x = (1/(torch.sqrt(1-scheduler.beta[0]))).to(device)*z-(temp*model(z,[0]))

        images.append(x.cpu())
        x = rearrange(x.squeeze(0),'c h w -> h w c').detach().cpu().numpy()
        #x = x.numpy()
        #plt.imshow(x)
        #plt.show()
        display_reverse(images)
        images = [] # Clear the images list for the next iteration


train(lr=2e-5,num_epochs=75,device='mps')

#train(checkpoint_path='checkpoints/ddpm_checkpoint',lr=2e-5,num_epochs=75)
inference('checkpoints/ddpm_checkpoint',device='cpu') # Run inference to generate samples from the trained model



