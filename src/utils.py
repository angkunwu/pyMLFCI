import pandas as pd
import numpy as np

# read data
def ReadAllData(Nx,Ny):
    pt = "~/QuarticCrossing/data/"
    alphas = np.linspace(0.78943,3.517548,100)
    atemp = np.round(alphas[0],5)
    file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(atemp) + ".csv"
    temp = pd.read_csv(file_path)
    temp_np = temp.to_numpy()
    N = int(temp_np.shape[1]/2)
    NBZ = temp_np.shape[0]/N
    NBZ = int(NBZ)
    y = np.zeros((400,1),dtype=bool)
    for k in range(100):
        atemp = np.round(alphas[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A"+ str(atemp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k] = temp.IsFCI[0]
    alpha = 0.78943
    c0s = np.linspace(-0.8,0.8,100)
    for k in range(100):
        c0temp = np.round(c0s[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k+100] = temp.IsFCI[0]
    alpha = 2.1325
    c0s = np.linspace(-0.6,0.6,100)
    for k in range(100):
        c0temp = np.round(c0s[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k+200] = temp.IsFCI[0]
    alpha = 3.517548
    c0s = np.linspace(-0.5,0.5,100)
    for k in range(100):
        c0temp = np.round(c0s[k],5)
        file_path = pt + "EDQBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        y[k+300] = temp.IsFCI[0]
    # convert y from a column-vector to 1d array
    y = y.flatten()
    
    train_data = np.zeros((N*N*NBZ,400),dtype=complex)
    for k in range(100):
        atemp = np.round(alphas[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(atemp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k] = temp_np_comp.reshape(-1) # convert into 1d array
    alpha = 0.78943
    c0s = np.linspace(-0.8,0.8,100)
    for k in range(100):
        c0temp = np.round(c0s[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k+100] = temp_np_comp.reshape(-1) # convert into 1d array
    alpha = 2.1325
    c0s = np.linspace(-0.6,0.6,100)
    for k in range(100):
        c0temp = np.round(c0s[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k+200] = temp_np_comp.reshape(-1) # convert into 1d array
    alpha = 3.517548
    c0s = np.linspace(-0.5,0.5,100)
    for k in range(100):
        c0temp = np.round(c0s[k],5)
        file_path = pt + "QBCPFFNx" + str(Nx) + "Ny" + str(Ny) + "A" + str(alpha) + "c0" + str(c0temp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,k+300] = temp_np_comp.reshape(-1) # convert into 1d array

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