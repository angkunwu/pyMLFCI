# In this code, we plan to use the training data to train a neural network classifier.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score,KFold

from src import utils # read data fun

train_data, y = utils.ReadAllData(3,5)


pt = "~/QuarticCrossing/data/"
alphas = np.linspace(0.78943,3.517548,100)
atemp = np.round(alphas[0],5)
file_path = pt + "QBCPFFNx3Ny5A" + str(atemp) + ".csv"
temp = pd.read_csv(file_path)
temp_np = temp.to_numpy()

N = int(temp_np.shape[1]/2)
NBZ = temp_np.shape[0]/N
NBZ = int(NBZ)
# initialize the y values as 100 by 1 matrix with boolean values
y = np.zeros((100,1),dtype=bool)
for k in range(100):
    atemp = np.round(alphas[k],5)
    file_path = pt + "EDQBCPFFNx3Ny5A" + str(atemp) + ".csv"
    temp = pd.read_csv(file_path)
    y[k] = temp.IsFCI[0]

Samples = np.sum(y)

#initialize training data as N*NBZ by 100 matrix with complex zeros
train_data = np.zeros((N*N*NBZ,y.shape[0]),dtype=complex)
CurInd = 0
for k in range(y.shape[0]):
    atemp = np.round(alphas[k],5)
    file_path = pt + "QBCPFFNx3Ny5A" + str(atemp) + ".csv"
    temp = pd.read_csv(file_path)
    temp_np = temp.to_numpy()
    temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
    train_data[:,k] = temp_np_comp.reshape(-1) # convert into 1d array





train_data_real = train_data.real
train_data_imag = train_data.imag
train_data_combined = np.concatenate((train_data_real, train_data_imag), axis=0)

train_data_abs = np.abs(train_data)
train_data_angle = np.angle(train_data)
train_data_combined = np.concatenate((train_data_angle, train_data_abs), axis=0)

X = train_data_combined.T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

mlp = MLPClassifier(
    hidden_layer_sizes = (50,50),
    max_iter = 100,
    alpha=1e-6,
    solver='lbfgs',#'sgd',
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)

mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

cv_scores = cross_val_score(mlp, X, y, cv=3)

print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: ", np.mean(cv_scores))





cv = KFold(n_splits=5, random_state=100, shuffle=True)
cv_scores = cross_val_score(mlp, X, y, cv=cv)
sum(cv_scores)/5

cvs = np.zeros((10,3))
for trial in range(10):
    cv = KFold(n_splits=5, random_state=100, shuffle=True)
    cv_scores = cross_val_score(mlp, X, y, cv=cv)
    cvs[trial,0] = sum(cv_scores)/5
    cvs[trial,1] = max(cv_scores)
    cvs[trial,2] = min(cv_scores)
    print(trial)

sum(cvs[:,0])/10
sum(cvs[:,1])/10
sum(cvs[:,2])/10
































import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data = torch.from_numpy(X_train)
test_data = torch.from_numpy(X_test)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
#device = "cpu"
# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

def train(dataloader, model, loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")