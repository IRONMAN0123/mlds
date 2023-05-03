import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def kernel(point, xmat, k):   
   m,n = np.shape(xmat)    
   weights = np.mat(np.eye((m)))   
   for j in range(m):      
     diff = point - X[j]  
     print("Point",point)
     print("Diff",diff)  
     weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))  
     print("Weights",weights)  
   return weights
def localWeight(point, xmat, ymat, k):    
  wei = kernel(point,xmat,k)    
  W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))   
  print("W",W) 
  return W
def localWeightRegression(xmat, ymat, k):    
  m,n = np.shape(xmat)    
  ypred = np.zeros(m)   
  for i in range(m):        
     ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)   
  return ypred
# load data points
data = pd.read_csv('C:\MLDS CSV FILES\hotel-bill1.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
#preparing and add 1 in bill
mbill = np.mat(bill)
print("MBILL",mbill)
mtip = np.mat(tip)
print("Mtip",mtip)
m= np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T))
print("X",X)
#set k here
ypred = localWeightRegression(X,mtip,2)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='green')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show();
