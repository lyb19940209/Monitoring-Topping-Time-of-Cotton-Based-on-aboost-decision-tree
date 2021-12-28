import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from scipy.io import savemat
#创造数据集
from sklearn.tree import DecisionTreeRegressor
spectrum1=loadmat('spectrum1.mat')
yelusu1=loadmat('yelusu1.mat')
yemoist1=loadmat('yemoist1.mat')
zhugao1= loadmat('zhugao1.mat')
hualei1=loadmat('hualei1.mat')
guozhi1=loadmat('guozhi1.mat')
rng = np.random.RandomState(1)
X=spectrum1['spectrum1']
y=guozhi1['guozhi1']
X=np.array(X)
X1=np.array(X[:,0])
X2=np.array(X[:,1])
X3=np.array(X[:,2])
X4=np.array(X[:,3])
#X=np.array([0.5*(200*(X4-X1)-200*(X2+X1))])
X=np.array([X4/X3-1])
X=np.transpose(X)
y=np.transpose(y).tolist()
#y=list(y)
Xtrain=X[0:600]
Ytrain=y[0:600]
Xtest=X[601:721]
Ytest=y[601:721]
#训练回归模
regr_1=DecisionTreeRegressor(max_depth=6,min_samples_split=12,min_weight_fraction_leaf=0.02,max_leaf_nodes=15)
#
#Xtrain.reshape((1,-1))
#Xtrain=Xtrain.reshape([-1,1])
regr_1.fit(Xtrain, Ytrain)
#预测结果
#Xtest=Xtest.reshape([-1,1])
y_1 = regr_1.predict(Xtest)
y_2 = regr_1.predict(Xtrain)
Ytest=np.array(Ytest)
Ytest1=Ytest.flatten()
Ytrain=np.array(Ytrain)
Ytrain1=Ytrain.flatten()
###########################################################
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
###########################################################
m11=sum((y_2-Ytrain1))
m11=m11*m11
n11=len(Ytest)
RMSEP22=math.sqrt(m11/n11)
R11=np.corrcoef(y_2, Ytrain1)
R22=1-m11/sum((Ytrain1-np.mean(Ytrain1)))
RPD2=1/(np.sqrt(1-R22))

#绘制可视化图形
plt.figure()
plt.plot(Ytest, c="k", label="testing samples")
plt.plot(y_1, c="g", label="DTR", linewidth=2) #
#plt.plot(y_2, c="r", label="ABR", linewidth=2) #
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
guozhiyucede=[y_2,y_1]
savemat('guozhiyucede1234.mat', {"guozhiyucede1234":guozhiyucede})
