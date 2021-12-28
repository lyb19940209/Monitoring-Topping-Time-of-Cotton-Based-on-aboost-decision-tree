function svm = Main_SVR(X,Y,ker,C,e)
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

% ------------------------------------------------------------%
% ѵ��֧��������

tic
svm = svmTrain('svr_epsilon',X,Y,ker,C,e);
t_train = toc;

% svm  ֧��������(�ṹ�����)
% the following fields:
%   type - ֧������������  {'svc_c','svc_nu','svm_one_class','svr_epsilon','svr_nu'}
%   ker - �˲���
%   x - ѵ������,d��n�ľ���,nΪ��������,dΪ����ά��
%   y - ѵ��Ŀ��,1��n�ľ���,nΪ��������
%   a - �������ճ���,1��n�ľ���

% ------------------------------------------------------------%
% Ѱ��֧������

%a = svm.a;
%epsilon = 1e-8;                     % ���"����ֵ"С�ڴ�ֵ����Ϊ��0
%i_sv = find(abs(a)>epsilon);        % ֧�������±�,�����abs(a)�����ж�
%plot(X(i_sv),Y(i_sv),'ro');