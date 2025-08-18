import numpy as np
import matplotlib.pyplot as plt
from src import utils

Nx, Ny = 4,6 #3, 5
#train_data, y = utils.ReadAllData(Nx, Ny)  # the train_data
train_data, y = utils.ReadAllData(Nx, Ny, 
                                   alphas = np.linspace(0.35,3.55,700),
                                   c0s1 = np.linspace(-1.0,1.0,100),
                                   c0s2 = np.linspace(-0.7,0.7,100),
                                   c0s3 = np.linspace(-0.5,0.5,100))  # the train_data
N = Nx * Ny
alphas = np.linspace(0.35,3.55,700)

indices = [1,10,20,28,29,50,100,150,200,226,227,230,240,245,249,250,300,350,400,407,408,409,410,450,500,530,533,534,540,544]
plt.figure()
plt.scatter(np.arange(1,len(indices)+1),y[indices])
plt.show()
