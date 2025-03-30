import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read .csv file and return a pandas dataframe
pt = "~/QuarticCrossing/data/"
alphas = np.linspace(0.78943,3.517548,100)
atemp = np.round(alphas[0],5)
#file_path = pt + "QBCPFFNx3Ny5A2.csv"
file_path = pt + "QBCPFFNx3Ny5A" + str(atemp) + ".csv"
test = pd.read_csv(file_path)

test_np = test.to_numpy()
#get the shape of the test_np
print(test_np.shape)
N = int(test_np.shape[1]/2)
NBZ = test_np.shape[0]/test_np.shape[1]
NBZ = int(NBZ)
# make a new matrix where the elements are complex numbers, whose real parts are the first N elements of the test_np and the imaginary parts are the last N elements of the test_np
test_np_comp = test_np[:,0:N] + 1j*test_np[:,N:2*N]
#reshape the test_np_comp to a 1D data as machine learning training data
test_np_1d = test_np_comp.reshape(-1,1)
test_np_1d2 = test_np_comp.reshape(-1)
# what's the difference between test_np_1d and test_np_1d2?



# divde NBZ by 2 and get integer value
BZind = int(NBZ/2)+1

FFmiddle = test_np_comp[((BZind-1)*N):(BZind*N),:]
#FFmiddle = test_np_comp[0:N,:]
# get the phase (Imaginary part of the log) of each element of the FFmiddle
FFmiddle_phase = np.angle(FFmiddle)
##FFmiddle_phase2 =np.imag(np.log(FFmiddle))

# plot the test_np as a heatmap
### %%
plt.figure()
plt.imshow(FFmiddle_phase, cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
plt.show()



# convert the julia code to python code α̃s = collect(LinRange(0.78943,3.517548,100)) α̃ = round(α̃s[k],digits=5)
alphas = np.linspace(0.78943,3.517548,100)
atemp = np.round(alphas[1],5)
for k in range(100):
    atemp = np.round(alphas[k],5)
    file_path = pt + "QBCPFFNx3Ny5A" + str(atemp) + ".csv"
    test = pd.read_csv(file_path)
    print(k)