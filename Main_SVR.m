function svm = Main_SVR(X,Y,ker,C,e)
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

% ------------------------------------------------------------%
% 训练支持向量机

tic
svm = svmTrain('svr_epsilon',X,Y,ker,C,e);
t_train = toc;

% svm  支持向量机(结构体变量)
% the following fields:
%   type - 支持向量机类型  {'svc_c','svc_nu','svm_one_class','svr_epsilon','svr_nu'}
%   ker - 核参数
%   x - 训练样本,d×n的矩阵,n为样本个数,d为样本维数
%   y - 训练目标,1×n的矩阵,n为样本个数
%   a - 拉格朗日乘子,1×n的矩阵

% ------------------------------------------------------------%
% 寻找支持向量

%a = svm.a;
%epsilon = 1e-8;                     % 如果"绝对值"小于此值则认为是0
%i_sv = find(abs(a)>epsilon);        % 支持向量下标,这里对abs(a)进行判定
%plot(X(i_sv),Y(i_sv),'ro');
