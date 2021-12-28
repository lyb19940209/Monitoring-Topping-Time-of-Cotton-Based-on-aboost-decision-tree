% 清理工作空间
clc;
clear;
%加载数据
load spectrum1
load guozhi1
load hualei1
load zhugao1
load yemoist1
load yelusu1
data4=spectrum1(:,4);
data3=spectrum1(:,3);
data2=spectrum1(:,2);
data1=spectrum1(:,1);
%dataall=[data1,data2,data3,data4];
dataall=data4/data3-1;
%Xtrain = spectrum1(1:600,3:4)';
%Ytrain = guozhi1(1,1:600);% 训练数据的真实输出矩阵
%Xtest = spectrum1(601:720,3:4)';
%Ytest = guozhi1(1,601:720);% 测试数据的真是输出矩阵
%[n,p]=size(Xtest);
%[n1,p1]=size(Xtrain);
Xtrain = dataall(1:600,:)';
Ytrain = zhugao1(1,1:600);
Xtest = dataall(601:720,:)';
Ytest = zhugao1(1,601:720);
[n,p]=size(Xtest);
[n1,p1]=size(Xtrain);

% 数据归一化
[train_x,IN] = mapminmax(Xtrain);
[train_y,OUT] = mapminmax(Ytrain);
test_x = mapminmax('apply',Xtest,IN);                                                       
test_y = Ytest;
% 创建神经网络
net = feedforwardnet([30,15]);
% 设置网络参数
net.trainParam.epochs = 10000000;
%net.trainParam.goal = 1e-8;
%net.trainParam.lr = 0.0000001;
%训练网络
net = train( net,train_x,train_y );
% 仿真测试
ysim = net( test_x );
xsim=net(train_x);
result=mapminmax('reverse' ,ysim,OUT); % 反归一化
xresult=mapminmax('reverse' ,xsim,OUT);
guozhibp1234=[xresult,result];
save('guozhibp1234','guozhibp1234')
% 绘图
figure;
plot( 1:120,test_y,'b-*',1:120,result,'r-o' );
legend( '真实值','预测值' );

% 计算60个测试样本的预测均方根误差
 %sum = 0;
% for i=1:48
%     sum=sum+(Ytest(i)-result(i))^2;
% end
% RMSEP=sqrt(sum/48);
% disp(RMSEP);
m=sum((test_y-mean(test_y)).^2);
m1=sum((result-mean(result)).^2);
m2=sum((test_y-result).^2);
RMSEP=sqrt(m2/n);%均方根误差
R=corrcoef(test_y, result);
R=R(1,2);
%R2=1-m/sum((test_y-mean(test_y)).^2);
R2=m1/m;
RPD=1/sqrt(1-R2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=sum((train_y-xresult).^2);
xRMSEP=sqrt(m1/p1);%均方根误差
R1=corrcoef(train_y, xresult);
R1=R1(1,2);
R21=1-m1/sum((train_y-mean(train_y)).^2);
RPD1=1/sqrt(1-R21);