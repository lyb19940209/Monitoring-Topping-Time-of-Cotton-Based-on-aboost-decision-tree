%-------------支持向量机回归-----------------------------
%使用Main_SVR函数训练得到svm结构体，
%使用svmSim函数仿真得到预测输出。
%-------------------------------------------------------

% ------------------------------------------------------------%
% 训练样本,d×n的矩阵,n为样本个数,d为样本维数
% 训练目标,1×n的矩阵,n为样本个数,值为期望输出
% ------------------------------------------------------------%
% 定义核函数及相关参数

%C = 100;                % 拉格朗日乘子上界
%e = 0.2;                % 不敏感损失函数的参数，Epsilon越大，支持向量越少

%ker = struct('type','linear');
%ker = struct('type','ploy','degree',5,'offset',1);
%ker = struct('type','gauss','width',0.6);
%ker = struct('type','tanh','gamma',1,'offset',0);

% ker - 核参数(结构体变量)
% the following fields:
%   type   - linear :  k(x,y) = x'*y
%            poly   :  k(x,y) = (x'*y+c)^d
%            gauss  :  k(x,y) = exp(-0.5*(norm(x-y)/s)^2)
%            tanh   :  k(x,y) = tanh(g*x'*y+c)
%   degree - Degree d of polynomial kernel (positive scalar).
%   offset - Offset c of polynomial and tanh kernel (scalar, negative for tanh).
%   width  - Width s of Gauss kernel (positive scalar).
%   gamma  - Slope g of the tanh kernel (positive scalar).

% ------------------------------------------------------------%
clear
close all
clc
load spectrum1
load guozhi1
load hualei1
load zhugao1
load yemoist1
load yelusu1
%Xtrain = spectrum1(1:600,3:4)';
%Ytrain(1,:) = hualei1(1,1:600);% 训练数据的真实输出矩阵
%Xtest = spectrum1(601:720,3:4)';
%Ytest = hualei1(1,601:720);% 测试数据的真是输出矩阵
%[n,p]=size(Xtest);
data4=spectrum1(:,4);
data3=spectrum1(:,3);
data2=spectrum1(:,2);
data1=spectrum1(:,1);
%data2=spectrum1(:,3:4);
%dataall=[data1,data2,data3,data4];
dataall=data4/data3-1;
Xtrain = dataall(1:600,:)';
Ytrain = zhugao1(1,1:600);
Xtest = dataall(601:720,:)';
Ytest = zhugao1(1,601:720);
[n,p]=size(Xtest);
[n1,p1]=size(Xtrain);
X=Xtrain;
Y=Ytrain;
Xt=Xtest;
Yt=Ytest;
%样本数据预处理
%X,Y为训练输入输出，Xt，Yt为验证输入与期望输出
% load('mp6spec.mat')
% load('propvals.mat')
% X= mp6spec.data;
% Y = propvals.data;
% X = X';
% Y = Y';
% Xt = X(:,1:60);
% Yt = Y(1,1:60);
% X = X(:,1:60);
% Y = Y(1,1:60);

%定义参数
%ker = struct('type','linear');
ker = struct('type','gauss','width',0.01);
e = 0.01;                         % 不敏感损失函数的参数，Epsilon越大，支持向量越少
C = 100;                         % 拉格朗日乘子上界

svm = Main_SVR(X,Y,ker,C,e);   %训练

%-------------------------------------------------------
% 测试输出
Yd = svmSim(svm,Xt);          
Yd1 = svmSim(svm,X);
figure(1);
plot(Yt,'b-');
hold on;
plot(Yd,'r-');
title('SVR验证输出','fontsize',12);
ylabel('函数输出','fontsize',12);
xlabel('样本','fontsize',12);

%计算总误差与平均误差
ERROR = Yt - Yd;
ERROR= Ytrain-Yd1;

PRESS1 = sum(ERROR.^2);
fprintf('样本均方根误差');
BMSEP = sqrt(PRESS1/length(Yt));  %均方根误差

R=corrcoef(Yd, Yt);
R=R(1, 2); % 预测集相关系数
m=sum((Yt-Yd).^2);
RMSEP=sqrt(m/n);%均方根误差

m=sum((Yt-mean(Yt)).^2);
m1=sum((Yd-mean(Yd)).^2);
m2=sum((Yd-Yt).^2);
RMSEP=sqrt(m2/n);%均方根误差
%R=corrcoef(test_y, result);
%R=R(1,2);
%R2=1-m/sum((test_y-mean(test_y)).^2);
R2=m1/m;

%R2=1-m/sum((Yt-mean(Yt)).^2);
%RPD=1/sqrt(1-R2);

[n1,p1]=size(Xtrain);
R1=corrcoef(Yd1, Ytrain);
R1=R1(1, 2); % 预测集相关系数
m1=sum((Ytrain-Yd1).^2);
RMSEP1=sqrt(m1/n);%均方根误差
R21=1-m/sum((Ytrain-mean(Ytrain)).^2);
RPD1=1/sqrt(1-R21);


mmin=min(Yt);
mmax=max(Yt)+0.5;
figure
plot(mmin:mmax,mmin:mmax,Yd,Yt,'o');
xlabel('实测值/°Brix','FontSize',10);%x轴
ylabel('预测值/°Brix','FontSize',10); %y轴
text(8.5,16.5,'R2=0.7056');
text(8.5,16,'RMSE=1.134');
text(8.5,15.5,'RPD=1.843');
