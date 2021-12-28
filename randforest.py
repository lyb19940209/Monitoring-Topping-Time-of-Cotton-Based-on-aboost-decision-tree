import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from scipy.io import savemat
from sklearn import datasets,ensemble
#创造数据集
spectrum1=loadmat('spectrum1.mat')
yelusu1=loadmat('yelusu1.mat')
yemoist1=loadmat('yemoist1')
zhugao1= loadmat('zhugao1.mat')
hualei1=loadmat('hualei1.mat')
guozhi1=loadmat('guozhi1.mat')
rng = np.random.RandomState(1)
X=spectrum1['spectrum1']
y=guozhi1['guozhi1']
X1=np.array(X[:,0])
X2=np.array(X[:,1])
X3=np.array(X[:,2])
X4=np.array(X[:,3])
X=np.array([X1,X2,X3,X4])
X=np.transpose(X)
y=np.transpose(y).tolist()
Xtrain=X[0:600]
Ytrain=y[0:600]
Xtest=X[601:721]
Ytest=y[601:721]
regr = RandomForestRegressor(n_estimators=1000,max_depth=6,min_samples_split=13,min_weight_fraction_leaf=0.01,max_leaf_nodes=20)
regr.fit(Xtrain, Ytrain)
y_1 = regr.predict(Xtest)
y_2 = regr.predict(Xtrain)
Ytest=np.array(Ytest)
Ytest1=Ytest.flatten()
Ytrain=np.array(Ytrain)
Ytrain1=Ytrain.flatten()
################################################
m1=np.sqrt(sum(pow((y_1-np.mean(y_1)),2)))
m2=np.sqrt(sum(pow(Ytest-np.mean(Ytest),2)))
print(sum(sum(pow((y_1-Ytest),2))))
print(len(y_1))
RMSEP=np.sqrt(sum(sum((pow((y_1-Ytest),2))))/len(y_1))
R1=np.corrcoef(y_1, Ytest1)
R2=m1/m2
#R2=1-m1/sum((Ytest-np.mean(Ytest1)))
RPD2=1/(np.sqrt(1-R2))
print('R2')
print(R2)
print('RMSEP')
print(RMSEP)
##################################################
m11=sum((y_2-Ytrain1))
m11=m11*m11
n11=len(Ytest)
RMSEP22=math.sqrt(m11/n11)
R11=np.corrcoef(y_2, Ytrain1)
R22=1-m11/sum((Ytrain1-np.mean(Ytrain1)))
RPD2=1/(np.sqrt(1-R22))
print('R11')
print(R11)
print('RMSEP22')
print(RMSEP22)
plt.figure()
plt.plot(Ytest, c="k", label="testing samples")
#plt.plot( y_1, c="g", label="DTR", linewidth=2) #决策树回归器
plt.plot(y_1, c="r", label="ABR", linewidth=2) #AdaBoost回归器
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
guozhiyucera=[y_2,y_1]
savemat('guozhiyucera1234.mat', {"guozhiyucera1234":guozhiyucera})