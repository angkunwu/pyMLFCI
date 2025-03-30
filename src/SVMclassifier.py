# In this code, we plan to use the training data to train a neural network classifier.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,KFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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
# change y from a column vector to a 1d array
y = y.reshape(-1)


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

svm_classifier = SVC(kernel='rbf',C=1, gamma=1) #'linear' 'poly' 'rbf' 'sigmoid'
cv = KFold(n_splits=5, random_state=42, shuffle=True)

cv_scores = cross_val_score(svm_classifier, X, y, cv=cv)

print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: ", np.mean(cv_scores))

train_data_real = train_data.real
train_data_imag = train_data.imag
train_data_combined = np.concatenate((train_data_real, train_data_imag), axis=0)

train_data_abs = np.abs(train_data)
train_data_angle = np.angle(train_data)
train_data_combined = np.concatenate((train_data_angle, train_data_abs), axis=0)

X = train_data_combined.T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

# how to see the performance of svm_classifier on the training set and test set?
svm_classifier.fit(X_train, y_train)
print("Training set score: %f" % svm_classifier.score(X_train, y_train))
print("Test set score: %f" % svm_classifier.score(X_test, y_test))


test = utils.EvaluateModel(svm_classifier, X_test, y_test)
test = utils.EvaluateModel(svm_classifier, X_train, y_train)



svm_classifier = SVC(kernel='linear',C=1)

cv = KFold(n_splits=5, random_state=100, shuffle=True)
cv_scores = cross_val_score(svm_classifier, X, y, cv=cv)
sum(cv_scores)/5

cvs = np.zeros((10,3))
for trial in range(10):
    cv = KFold(n_splits=5, random_state=100, shuffle=True)
    cv_scores = cross_val_score(svm_classifier, X, y, cv=cv)
    cvs[trial,0] = sum(cv_scores)/5
    cvs[trial,1] = max(cv_scores)
    cvs[trial,2] = min(cv_scores)
    print(trial)

sum(cvs[:,0])/10
sum(cvs[:,1])/10
sum(cvs[:,2])/10
