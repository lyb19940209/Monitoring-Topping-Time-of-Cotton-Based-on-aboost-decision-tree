% �������ռ�
clc;
clear;
%��������
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
%Ytrain = guozhi1(1,1:600);% ѵ�����ݵ���ʵ�������
%Xtest = spectrum1(601:720,3:4)';
%Ytest = guozhi1(1,601:720);% �������ݵ������������
%[n,p]=size(Xtest);
%[n1,p1]=size(Xtrain);
Xtrain = dataall(1:600,:)';
Ytrain = zhugao1(1,1:600);
Xtest = dataall(601:720,:)';
Ytest = zhugao1(1,601:720);
[n,p]=size(Xtest);
[n1,p1]=size(Xtrain);

% ���ݹ�һ��
[train_x,IN] = mapminmax(Xtrain);
[train_y,OUT] = mapminmax(Ytrain);
test_x = mapminmax('apply',Xtest,IN);                                                       
test_y = Ytest;
% ����������
net = feedforwardnet([30,15]);
% �����������
net.trainParam.epochs = 10000000;
%net.trainParam.goal = 1e-8;
%net.trainParam.lr = 0.0000001;
%ѵ������
net = train( net,train_x,train_y );
% �������
ysim = net( test_x );
xsim=net(train_x);
result=mapminmax('reverse' ,ysim,OUT); % ����һ��
xresult=mapminmax('reverse' ,xsim,OUT);
guozhibp1234=[xresult,result];
save('guozhibp1234','guozhibp1234')
% ��ͼ
figure;
plot( 1:120,test_y,'b-*',1:120,result,'r-o' );
legend( '��ʵֵ','Ԥ��ֵ' );

% ����60������������Ԥ����������
 %sum = 0;
% for i=1:48
%     sum=sum+(Ytest(i)-result(i))^2;
% end
% RMSEP=sqrt(sum/48);
% disp(RMSEP);
m=sum((test_y-mean(test_y)).^2);
m1=sum((result-mean(result)).^2);
m2=sum((test_y-result).^2);
RMSEP=sqrt(m2/n);%���������
R=corrcoef(test_y, result);
R=R(1,2);
%R2=1-m/sum((test_y-mean(test_y)).^2);
R2=m1/m;
RPD=1/sqrt(1-R2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=sum((train_y-xresult).^2);
xRMSEP=sqrt(m1/p1);%���������
R1=corrcoef(train_y, xresult);
R1=R1(1,2);
R21=1-m1/sum((train_y-mean(train_y)).^2);
RPD1=1/sqrt(1-R21);