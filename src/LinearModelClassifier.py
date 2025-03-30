import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from src import utils # read data fun

train_data, y = utils.ReadAllData(3,5)

# check the memory usage of train_data in MB
train_data.nbytes/1024**2

# plot the real part of train_data as a matrix
plt.imshow(train_data.real[14200:14350,1:100])
plt.show()


train_data_real = train_data.real
train_data_imag = train_data.imag
train_data_combined = np.concatenate((train_data_real, train_data_imag), axis=0)

X = train_data_combined.T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

# Linear regression
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
# predict
reg.score(X_train, y_train)
reg.score(X_test, y_test)
#The best possible score is 1.0 and it can be negative 
# (because the model can be arbitrarily worse).
reg.predict(X_test)



# logistic regression
from sklearn.linear_model import LogisticRegression

X = train_data_combined.T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

#LogistReg = LogisticRegression(penalty="l1", tol=0.005, solver="saga")
LogistReg = LogisticRegression(penalty="l2", tol=0.1, solver="saga")
LogistReg.fit(X_train, y_train)

LogistReg.score(X_train, y_train)
LogistReg.score(X_test, y_test)

LogistReg.predict(X_test)

def EvaluateModel(MODEL, X, y):
    # Get predictions
    y_pred = MODEL.predict(X)
    # Compute accuracy
    accuracy = np.sum(y_pred == y) / np.size(y)
    # Compute confusion matrix components
    TP = np.sum((y_pred == 1) & (y == 1))  # True Positives
    FP = np.sum((y_pred == 1) & (y == 0))  # False Positives
    TN = np.sum((y_pred == 0) & (y == 0))  # True Negatives
    FN = np.sum((y_pred == 0) & (y == 1))  # False Negatives
    F1score = 2*TP/(2*TP + FP + FN)
    return accuracy, F1score

test = EvaluateModel(LogistReg, X_test, y_test)

test = EvaluateModel(LogistReg, X_train, y_train)


# cross validation
from sklearn.model_selection import cross_val_score, KFold

train_data_real = train_data.real
train_data_imag = train_data.imag
train_data_combined = np.concatenate((train_data_real, train_data_imag), axis=0)
X = train_data_combined.T

cv = KFold(n_splits=5, random_state=100, shuffle=True)
cv_scores = cross_val_score(LogistReg, X, y, cv=cv)
sum(cv_scores)/5

cvs = np.zeros((10,3))
for trial in range(10):
    cv = KFold(n_splits=5, random_state=100, shuffle=True)
    cv_scores = cross_val_score(LogistReg, X, y, cv=cv)
    cvs[trial,0] = sum(cv_scores)/5
    cvs[trial,1] = max(cv_scores)
    cvs[trial,2] = min(cv_scores)
    print(trial)

sum(cvs[:,0])/10
sum(cvs[:,1])/10
sum(cvs[:,2])/10



# Bayesian Ridge and Gaussian process are too expensive
