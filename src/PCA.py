import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import utils


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
train_data = np.zeros((N*N*NBZ,Samples),dtype=complex)
CurInd = 0
for k in range(100):
    if y[k]:
        atemp = np.round(alphas[k],5)
        file_path = pt + "QBCPFFNx3Ny5A" + str(atemp) + ".csv"
        temp = pd.read_csv(file_path)
        temp_np = temp.to_numpy()
        temp_np_comp = temp_np[:,0:N] + 1j*temp_np[:,N:2*N]
        train_data[:,CurInd] = temp_np_comp.reshape(-1) # convert into 1d array
        CurInd = CurInd + 1
#train_data_abs = np.abs(train_data)
#train_data_angle = np.angle(train_data)
#train_data = np.concatenate((train_data_angle, train_data_abs), axis=0)


################################################
# import new data
train_data, y = utils.ReadAllData(3,5)

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

#train_data, y = filterData(train_data,y)

Samples = np.size(y)

# mean of the training data
barx = np.mean(train_data,axis=1)

# Principal component analysis
XT = train_data - barx[:,None]
X = XT.conj().T
S = np.dot(X,XT)/Samples
eigenvalues, eigenvectors = np.linalg.eig(S)

np.dot(eigenvectors[:,68].conj(),eigenvectors[:,68])


normalvectors = np.zeros(XT.shape,dtype=complex)
for k in range(Samples):
    normalvectors[:,k] = np.dot(XT,eigenvectors[:,k])/np.sqrt(eigenvalues[k]*Samples)

np.dot(normalvectors[:,30].conj(),normalvectors[:,30])

def obtainExpansion(XT,normalvectors):
    Samples = XT.shape[1]
    Expansion = np.zeros((Samples,Samples),dtype=complex)
    for k in range(Samples):
        Expansion[:,k] = np.dot(XT.conj().T,normalvectors[:,k])
    return Expansion

expansions = obtainExpansion(XT,normalvectors)

test = np.real(eigenvalues)
xs = np.arange(1,Samples)
# scatter plot test, with log scale in y axis
plt.figure()
plt.scatter(np.arange(1,Samples+1),test)
plt.yscale('log')
plt.show()



plt.figure()
plt.scatter(np.arange(1,XT.shape[0]+1),np.angle(XT[:,0]))
plt.scatter(np.arange(1,normalvectors.shape[0]+1),np.angle(normalvectors[:,0]))
#plt.yscale('log')
plt.show()


plt.figure()
plt.scatter(np.arange(1,expansions.shape[0]+1),np.abs(expansions[3,:]))
plt.show()

# plot all samples' expansions, each curve is np.abs(expansions[k,:])
plt.figure()
for k in range(20):
    plt.scatter(np.arange(1,expansions.shape[0]+1),np.abs(expansions[(k-1)*20,:]))
# set xlabel and ylabel
plt.xlabel(r'expansion order $j$',fontsize=12)
plt.ylabel(r'amplitude $\mathbf{u}_j\mathbf{x}_n$',fontsize=12)
plt.yscale('log')
# set x, y ticks size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0,200)
#plt.yscale('log')
plt.show()
# save to a .pdf file to the Desktop
#pt = "/Users/angkunwu/Desktop/"
#plt.savefig(pt+'expansion.pdf',format='pdf')


'''
#take real part of the training data
train_data_real = np.real(train_data)
# using PCA to reduce the dimension of the training data under pandas
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_data_real)
principalDf = pd.DataFrame(data = principalComponents,
             columns = ['principal component 1', 'principal component 2'])
# plot the reduced data
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
ax.grid()
plt.show()
'''
# generate a function that convert the 1D data back to 2D data
def convert1Dto2D(data1d,N,NBZ):
    data2d = np.zeros((N*NBZ,N),dtype=complex)
    for k in range(NBZ):
        data2d[(k*N):((k+1)*N),:] = data1d[(k*N*N):((k+1)*N*N)].reshape(N,N)
    return data2d
'''
test = XT[:,34]
FFfull = convert1Dto2D(test,N,NBZ)
#FFfull = convert1Dto2D(normalvectors[:,0],N,NBZ)
BZind = int(NBZ/2)+1
FFmiddle = FFfull[((BZind-1)*N):(BZind*N),:]
FFmiddle_phase = np.angle(FFmiddle)
### %%
#plt.figure()
#plt.imshow(FFmiddle_phase, cmap='RdBu', interpolation='nearest')
#plt.colorbar() # add color bar
#plt.show()
'''
def approxFF(xn,normalvectors,ExpOrder):
    xnT = xn.conj().T
    totdim = normalvectors.shape[0]
    res = np.zeros(totdim,dtype=complex)
    for k in range(ExpOrder):
        res = res + np.dot(xnT,normalvectors[:,k])*normalvectors[:,k]
    return res

#np.dot(XT[:,34].conj(),XT[:,34])
#np.dot(test.conj(),XT[:,34])

# save the normalvectors to a .csv file
sample = 34 # 34 49
test = approxFF(XT[:,sample],normalvectors,9) #XT[:,34] 
test = test + barx

FFfull = convert1Dto2D(test,N,NBZ)
FFfull_real = np.real(FFfull)
FFfull_imag = np.imag(FFfull)

headerRe = ["K"+str(k)+"Re" for k in range(1,N+1)]
headerIm = ["K"+str(k)+"Im" for k in range(1,N+1)]
header = headerRe + headerIm
outMat = np.zeros((FFfull_real.shape[0],2*N))
outMat[:,0:N] = FFfull_real
outMat[:,N:2*N] = FFfull_imag
df = pd.DataFrame(outMat,columns=header)

# save the df to a .csv file
pt = "~/pyMLFCI/data/"
file_path = pt + "PCA1.csv"
df.to_csv(file_path,index=False)

'''
# the y axis label is block by the figure size, how to fix it?
plt.figure()
plt.scatter(np.arange(1,XT.shape[0]+1),np.angle(test-train_data[:,sample]))
plt.xlabel(r'index $k$',fontsize=12)
#plt.ylabel(r'$|\tilde{\mathbf{x}}_{nk}-\mathbf{x}_{nk}|$',fontsize=12)
plt.ylabel(r'$\theta[\tilde{\mathbf{x}}_{nk}-\mathbf{x}_{nk}]$',fontsize=12)
# set x, y ticks size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# adjust the margin of the figure
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
plt.show()
#pt = "/Users/angkunwu/Desktop/"
#plt.savefig(pt+'fig8.png',format='png',dpi=300)
'''

sample = 49 # 34 49

def outputPCAFF(sample, expOrder):
    atemp = np.round(alphas[sample],5)
    test = approxFF(XT[:,sample],normalvectors,expOrder) #XT[:,34] 
    test = test + barx
    FFfull = convert1Dto2D(test,N,NBZ)
    FFfull_real = np.real(FFfull)
    FFfull_imag = np.imag(FFfull)

    headerRe = ["K"+str(k)+"Re" for k in range(1,N+1)]
    headerIm = ["K"+str(k)+"Im" for k in range(1,N+1)]
    header = headerRe + headerIm
    outMat = np.zeros((FFfull_real.shape[0],2*N))
    outMat[:,0:N] = FFfull_real
    outMat[:,N:2*N] = FFfull_imag
    df = pd.DataFrame(outMat,columns=header)

    # save the df to a .csv file
    pt = "~/pyMLFCI/data/"
    file_path = pt + "FFPCA"+ str(atemp) + "Order" + str(expOrder) + ".csv"
    df.to_csv(file_path,index=False)
    return

for k in range(30):
    outputPCAFF(sample, k+1)