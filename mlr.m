%b为参数，bint回归系数的区间估计，r为残差，
%rint为置信区间，stats用于回归模型检验
%stats=[r2(相关系数，r2越接近1说明数据拟合程度越好),F(F统计量用于检验模型是
%否通过检验。通过查F分布表，如果F>F分布表中对应的值，则通过检验),p(针对检验
%原假设(即:变量系数为零)成立的概率。当p<0.05(alpha默认值),说明成立的可能性
%很小，不到5%，最好是0,p<alpha时拒绝H0,即拒绝零假设，认为回归方程中至少有一
%个变量的系数不为零回归模型成立),残差方差]
%检验回归模型：相关系数r^2=stats(1,1)越接近1回归方程越显著
%F=stats(1,2)值越大回归方程越显著
%p=stats(1,3)<0.05时回归模型成立
%stats(1,4)残差的方差，越小越好
function [RMSEP,R,R2,RPD] = mlr(xtrain,ytrain,xtest,ytest)
pz=[xtrain,ytrain];
[row,col]=size(pz);
x=pz(:,1:col-1);
y=pz(:,end);
[m,n]=size(x);
X=[ones(m,1),x];
[b,bint,r,rint,stats]=regress(y,X);
% stats
% disp('回归模型检验:');
% if (stats(1,3)<0.05)&(stats(1,1)>0.6)
%    disp('回归方程显著,模型成立');
% end
test=[xtest,ytest];
[row1,col1]=size(test);
aa1=test(:,1:col1-1);%自变量	
aa1 = aa1';
xish = b(2:col1)';
bb1=xish; %回归系数
ch0 = b(1);%常数项
c=repmat(ch0,1,row1); %行向量
s=bb1*aa1+c;
yy=s';
d=test(:,end);%实测值
m=sum((d-yy).^2);
RMSEP=sqrt(m/row1);%均方根误差
R=corrcoef(d,yy);
R=R(1,2);
R2=1-m/sum((d-mean(ytest)).^2);
RPD=1/sqrt(1-R2);
mmin=min(ytest)-1;
mmax=max(ytest)+1;
% figure
% plot(mmin:mmax,mmin:mmax,d,yy,'o');
% xlabel('实测值/°Brix','FontSize',10);%x轴
% ylabel('预测值/°Brix','FontSize',10); %y轴
end

