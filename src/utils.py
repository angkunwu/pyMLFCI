import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset

# read data
def ReadAllData(Nx,Ny,
                alphas = np.linspace(0.78943,3.517548,100),
                c0s1 = np.linspace(-0.8,0.8,100),
                c0s2 = np.linspace(-0.6,0.6,100),
                c0s3 = np.linspace(-0.5,0.5,100),
                folder = "~/QuarticCrossing/data/"
                ):
    pt = folder + str(Nx) + str(Ny) + "data/" 
    #alphas = np.linspace(0.78943,3.517548,100)
    atemp = np.round(alphas[0],5)
    file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(atemp) + ".csv"
    temp = pd.read_csv(file_path)
    temp_np = temp.to_numpy()
    N = int(temp_np.shape[1]/2)
    NBZ = temp_np.shape[0]/N
    NBZ = int(NBZ)
    Samples = len(alphas) + len(c0s1) + len(c0s2) + len(c0s3)
    y = np.zeros((Samples,1),dtype=bool)
    for k in range(len(alphas)):
        atemp = np.round(alphas[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A"+ str(atemp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k] = temp.IsFCI[0]
    alpha = 0.78943
    #c0s1 = np.linspace(-0.8,0.8,100)
    for k in range(len(c0s1)):
        c0temp = np.round(c0s1[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k+len(alphas)] = temp.IsFCI[0]
    alpha = 2.1325
    #c0s2= np.linspace(-0.6,0.6,100)
    for k in range(len(c0s2)):
        c0temp = np.round(c0s2[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k+len(alphas)+len(c0s1)] = temp.IsFCI[0]
    alpha = 3.517548
    #c0s = np.linspace(-0.5,0.5,100)
    for k in range(len(c0s3)):
        c0temp = np.round(c0s3[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k+len(alphas)+len(c0s1)+len(c0s2)] = temp.IsFCI[0]
    # convert y from a column-vector to 1d array
    y = y.flatten()
    
    train_data = np.zeros((N*N*NBZ,Samples),dtype=complex)
    for k in range(len(alphas)):
        atemp = np.round(alphas[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(atemp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k] = temp_np_comp.reshape(-1) # convert into 1d array
    alpha = 0.78943
    #c0s = np.linspace(-0.8,0.8,100)
    for k in range(len(c0s1)):
        c0temp = np.round(c0s1[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k+len(alphas)] = temp_np_comp.reshape(-1) # convert into 1d array
    alpha = 2.1325
    #c0s = np.linspace(-0.6,0.6,100)
    for k in range(len(c0s2)):
        c0temp = np.round(c0s2[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k+len(alphas)+len(c0s1)] = temp_np_comp.reshape(-1) # convert into 1d array
    alpha = 3.517548
    #c0s = np.linspace(-0.5,0.5,100)
    for k in range(len(c0s3)):
        c0temp = np.round(c0s3[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k+len(alphas)+len(c0s1)+len(c0s2)] = temp_np_comp.reshape(-1) # convert into 1d array

    return train_data, y



def EvaluateModel(MODEL, X, y):
    # Get predictions
    y_pred = MODEL.predict(X)
    # Compute accuracy
    accuracy = np.sum(y_pred == y) / np.size(y)
    # Compute confusion matrix components
    TP = np.sum((y_pred == 1) & (y == 1))  # True Positives
    FP = np.sum((y_pred == 1) & (y == 0))  # False Positives
    print("False Positives: ", FP/np.size(y))
    TN = np.sum((y_pred == 0) & (y == 0))  # True Negatives
    FN = np.sum((y_pred == 0) & (y == 1))  # False Negatives
    print("False Negatives: ", FN/np.size(y))
    F1score = 2*TP/(2*TP + FP + FN)
    return accuracy, F1score

# generate a function that convert the 1D data back to 2D data
def convert1Dto2D(data1d,N,NBZ):
    #data2d = np.zeros((N,N*NBZ),dtype=complex)
    data2d = np.zeros((N*NBZ,N),dtype=complex)
    for k in range(NBZ):
        #data2d[:,k*N:(k+1)*N] = data1d[k*N*N:(k+1)*N*N].reshape(N,N)
        data2d[(k*N):((k+1)*N),:] = data1d[(k*N*N):((k+1)*N*N)].reshape(N,N)
    # transpose the data2d
    data2d = data2d.T
    return data2d


class FormFactorDataset(Dataset):
    def __init__(self, data, labels, Nx, Ny, NBZ, transform=None):
        """
        Args:
            data(numpy.ndarray): 2D array where each column is a flattened image.
            labels (numpy.ndarray): Labels for corresponding each image
            Nx (int): Number of rows in the BZ.
            Ny (int): Number of columns in the BZ.
            NBZ (int): Number of BZ.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.Nx = Nx
        self.Ny = Ny
        self.NBZ = NBZ

    def __len__(self):
        return self.data.shape[1]  # number of samples

    def __getitem__(self, idx):
        data1d = self.data[:, idx]
        data_real = data1d.real
        data_imag = data1d.imag
        data_combined = np.concatenate((data_real, data_imag), axis = 0)
        # Normalize the data (optional, depending on your model)
        data_combined = (data_combined) / 2.0  # Scale to [-0.5, 0.5]
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data_combined, dtype=torch.float32)
        # Apply any additional transformations
        if self.transform:
            data_tensor = self.transform(data_tensor)
        label = self.labels[idx]
        label_tensor = torch.tensor(bool(label), dtype = torch.bool)
        return data_tensor, label_tensor


def filterData(train_data,y):
    Samples = np.sum(y)
    train_data_new = np.zeros((train_data.shape[0],Samples),dtype=complex)
    ynew = np.zeros((Samples,1),dtype=bool)
    CurInd = 0
    for k in range(train_data.shape[1]):
        if y[k]:
            train_data_new[:,CurInd] = train_data[:,k]
            ynew[CurInd] = y[k]
            CurInd = CurInd + 1
    ynew = ynew.flatten()
    return train_data_new, ynew