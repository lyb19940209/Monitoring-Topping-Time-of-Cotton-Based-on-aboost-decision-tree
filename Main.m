%-------------֧���������ع�-----------------------------
%ʹ��Main_SVR����ѵ���õ�svm�ṹ�壬
%ʹ��svmSim��������õ�Ԥ�������
%-------------------------------------------------------

% ------------------------------------------------------------%
% ѵ������,d��n�ľ���,nΪ��������,dΪ����ά��
% ѵ��Ŀ��,1��n�ľ���,nΪ��������,ֵΪ�������
% ------------------------------------------------------------%
% ����˺�������ز���

%C = 100;                % �������ճ����Ͻ�
%e = 0.2;                % ��������ʧ�����Ĳ�����EpsilonԽ��֧������Խ��

%ker = struct('type','linear');
%ker = struct('type','ploy','degree',5,'offset',1);
%ker = struct('type','gauss','width',0.6);
%ker = struct('type','tanh','gamma',1,'offset',0);

% ker - �˲���(�ṹ�����)
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
%Ytrain(1,:) = hualei1(1,1:600);% ѵ�����ݵ���ʵ�������
%Xtest = spectrum1(601:720,3:4)';
%Ytest = hualei1(1,601:720);% �������ݵ������������
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
%��������Ԥ����
%X,YΪѵ�����������Xt��YtΪ��֤�������������
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

%�������
%ker = struct('type','linear');
ker = struct('type','gauss','width',0.01);
e = 0.01;                         % ��������ʧ�����Ĳ�����EpsilonԽ��֧������Խ��
C = 100;                         % �������ճ����Ͻ�

svm = Main_SVR(X,Y,ker,C,e);   %ѵ��

%-------------------------------------------------------
% �������
Yd = svmSim(svm,Xt);          
Yd1 = svmSim(svm,X);
figure(1);
plot(Yt,'b-');
hold on;
plot(Yd,'r-');
title('SVR��֤���','fontsize',12);
ylabel('�������','fontsize',12);
xlabel('����','fontsize',12);

%�����������ƽ�����
ERROR = Yt - Yd;
ERROR= Ytrain-Yd1;

PRESS1 = sum(ERROR.^2);
fprintf('�������������');
BMSEP = sqrt(PRESS1/length(Yt));  %���������

R=corrcoef(Yd, Yt);
R=R(1, 2); % Ԥ�⼯���ϵ��
m=sum((Yt-Yd).^2);
RMSEP=sqrt(m/n);%���������

m=sum((Yt-mean(Yt)).^2);
m1=sum((Yd-mean(Yd)).^2);
m2=sum((Yd-Yt).^2);
RMSEP=sqrt(m2/n);%���������
%R=corrcoef(test_y, result);
%R=R(1,2);
%R2=1-m/sum((test_y-mean(test_y)).^2);
R2=m1/m;

%R2=1-m/sum((Yt-mean(Yt)).^2);
%RPD=1/sqrt(1-R2);

[n1,p1]=size(Xtrain);
R1=corrcoef(Yd1, Ytrain);
R1=R1(1, 2); % Ԥ�⼯���ϵ��
m1=sum((Ytrain-Yd1).^2);
RMSEP1=sqrt(m1/n);%���������
R21=1-m/sum((Ytrain-mean(Ytrain)).^2);
RPD1=1/sqrt(1-R21);


mmin=min(Yt);
mmax=max(Yt)+0.5;
figure
plot(mmin:mmax,mmin:mmax,Yd,Yt,'o');
xlabel('ʵ��ֵ/��Brix','FontSize',10);%x��
ylabel('Ԥ��ֵ/��Brix','FontSize',10); %y��
text(8.5,16.5,'R2=0.7056');
text(8.5,16,'RMSE=1.134');
text(8.5,15.5,'RPD=1.843');
