import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import utils
from src import FormFactorFuns as FFFs

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

Nx, Ny = 3, 5
train_data, y = utils.ReadAllData(Nx,Ny)

alphas = np.linspace(0.78943,3.517548,100)
#alphas[49]
FormFactor1d = train_data[:,99]
N = Nx * Ny
NBZ = FormFactor1d.shape[0]/N/N
NBZ = int(NBZ)
FormFactor2d = convert1Dto2D(FormFactor1d,N,NBZ)

BZind = int(NBZ/2)+1
FFmiddle = FormFactor2d[:,((BZind-1)*N):(BZind*N)]
FFmiddle_phase = np.angle(FFmiddle)
FFmiddle_abs = np.abs(FFmiddle)

# plot all samples' expansions, each curve is np.abs(expansions[k,:])
plt.figure()
# plot FFmiddle_phase
plt.imshow(FFmiddle_abs, cmap='RdBu', interpolation='nearest')
plt.colorbar() # add color bar
#plt.yscale('log')
plt.show()


"""
test = WilsonLoopFull(Nx, Ny, FormFactors;etaxy=false)
testetaxy = WilsonLoopFull(Nx, Ny, FormFactors;etaxy=true)
sum(test)/2/pi
sum(testetaxy)/2/pi/2
test2 = FSoverlapFull(Nx, Ny, FormFactors)
"""
test = FFFs.WilsonLoopFull(Nx, Ny, FormFactor2d, etaxy=False)
sum(test)/2/np.pi
testetaxy = FFFs.WilsonLoopFull(Nx, Ny, FormFactor2d,etaxy=True)
sum(testetaxy)/2/np.pi/2

test2 = FFFs.FSoverlapFull(Nx, Ny, FormFactor2d)
sum(test2)/2/np.pi

plt.figure()
# scatter plot Berry curvature
plt.scatter(np.arange(1,16),-test,label=r'$F_{xy}$')
plt.scatter(np.arange(1,16),testetaxy/2,label=r'$\mathrm{tr} g_1$')
plt.scatter(np.arange(1,16),test2,label=r'$\mathrm{tr} g_2$')
plt.xlabel(r'$k$',fontsize=12)
plt.ylabel(r'$F_{xy}$',fontsize=12)
#plt.yscale('log')
plt.show()



# PCA
Samples = np.size(y)
barx = np.mean(train_data,axis=1)
XT = train_data - barx[:,None]
X = XT.conj().T
S = np.dot(X,XT)/Samples
eigenvalues, eigenvectors = np.linalg.eig(S)

plt.figure()
plt.scatter(np.arange(1,eigenvalues.shape[0]+1),np.abs(eigenvalues))
# set xlabel and ylabel
plt.xlabel(r'principal component $j$',fontsize=12)
plt.ylabel(r'eigenvalue $\lambda_j$',fontsize=12)
plt.yscale('log')
# set x, y ticks size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0,200)
plt.ylim(1e-14,100)
plt.show()


normalvectors = np.zeros(XT.shape,dtype=complex)
for k in range(Samples):
    normalvectors[:,k] = np.dot(XT,eigenvectors[:,k])/np.sqrt(eigenvalues[k]*Samples)

Fxys = np.zeros(Samples)
trg = np.zeros(Samples)
trg1 = np.zeros(Samples)
for k in range(Samples):
    Component2d = convert1Dto2D(normalvectors[:,k],N,NBZ)
    Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
    trgs = FFFs.FSoverlapFull(Nx, Ny, Component2d)
    trg1s = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
    Fxys[k] = sum(Bcurs)/2/np.pi
    trg[k] = sum(trgs)/2/np.pi
    trg1[k] = sum(trg1s)/2/np.pi/2

plt.figure()
plt.scatter(np.arange(1,Samples+1),Fxys,label=r'$F_{xy}$')
plt.scatter(np.arange(1,Samples+1),trg1,label=r'$\mathrm{tr} g_1$')
plt.scatter(np.arange(1,Samples+1),trg,label=r'$\mathrm{tr} g_2$')
plt.xlabel(r'principal component $j$',fontsize=12)
plt.ylabel(r'geometric quantities',fontsize=12)
plt.legend(fontsize=12,loc='upper right')
# set x, y ticks size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
# save to a .pdf file to the Desktop
#pt = "/Users/angkunwu/Desktop/"
#plt.savefig(pt+'expansion.pdf',format='pdf')




def obtainExpansion(XT,normalvectors):
    Samples = XT.shape[1]
    Expansion = np.zeros((Samples,Samples),dtype=complex)
    for k in range(Samples):
        Expansion[:,k] = np.dot(XT.conj().T,normalvectors[:,k])
    return Expansion

expansions = obtainExpansion(XT,normalvectors)

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
plt.show()

def approxFF(xn,normalvectors,ExpOrder):
    xnT = xn.conj().T
    totdim = normalvectors.shape[0]
    res = np.zeros(totdim,dtype=complex)
    for k in range(ExpOrder):
        res = res + np.dot(xnT,normalvectors[:,k])*normalvectors[:,k]
    return res

def allLocalMetric(FormFactor1D):
    Component2d = convert1Dto2D(FormFactor1D,N,NBZ)
    Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
    #trg2s = FFFs.FSoverlapFull(Nx, Ny, Component2d)
    trg1s = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
    #return sum(Bcurs)/2/np.pi, sum(trg1s)/2/np.pi/2, sum(trg2s)/2/np.pi
    return sum(Bcurs)/2/np.pi, sum(trg1s)/2/np.pi/2, np.std(Bcurs), np.std(trg1s)
# save the normalvectors to a .csv file
sample = 34 # 34 49
test = approxFF(XT[:,sample],normalvectors,100) #XT[:,34] # XT is subtracted by mean!!!!!
test = test + barx
allLocalMetric(test)

Component2d = convert1Dto2D(test,N,NBZ)
Bcurs = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
trg1s = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
Component2d = convert1Dto2D(train_data[:,sample],N,NBZ) # instead of XT[:,sample]!!!
Bcurs0 = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=False)
trg1s0 = FFFs.WilsonLoopFull(Nx, Ny, Component2d, etaxy=True)
#???? why negative
plt.figure()
plt.scatter(np.arange(1,15+1),-Bcurs0,label=r'$-F_{xy}$',s=30)
plt.scatter(np.arange(1,15+1),-Bcurs,label=r'$-\tilde{F}_{xy}$',s=5)
plt.scatter(np.arange(1,15+1),trg1s0/2,label=r'$\mathrm{tr} g$',s=30)
plt.scatter(np.arange(1,15+1),trg1s/2,label=r'$\mathrm{tr} \tilde{g}$',s=5)
plt.xlabel(r'Momentum Sector $K$',fontsize=12)
plt.ylabel(r'geometric quantities',fontsize=12)
plt.legend(fontsize=12,loc='upper right')
# set x, y ticks size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


orders = 400 #70
Fxys = np.zeros(orders)
trg = np.zeros(orders)
trg1 = np.zeros(orders)
stdFxys = np.zeros(orders)
stdtrg = np.zeros(orders)
for k in range(orders):
    test = approxFF(XT[:,sample],normalvectors,k)
    test = test + barx
    #Fxys[k], trg1[k], trg[k] = allLocalMetric(test)
    Fxys[k], trg1[k], stdFxys[k], stdtrg[k] = allLocalMetric(test)

plt.figure()
plt.scatter(np.arange(1,orders+1),-Fxys,label=r'$F_{xy}$',s=10)
plt.scatter(np.arange(1,orders+1),trg1,label=r'$\mathrm{tr} g$',s=10)
#plt.scatter(np.arange(1,orders+1),trg,label=r'$\mathrm{tr} g_2$',s=10)
plt.scatter(np.arange(1,orders+1),stdFxys,label=r'$\sigma_{F_{xy}}$',s=10)
plt.scatter(np.arange(1,orders+1),stdtrg,label=r'$\sigma_{\mathrm{tr} g}$',s=10)
plt.xlabel(r'expansions $M$',fontsize=12)
plt.ylabel(r'geometric quantities',fontsize=12)
plt.legend(fontsize=12,loc='upper right')
# set x, y ticks size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()