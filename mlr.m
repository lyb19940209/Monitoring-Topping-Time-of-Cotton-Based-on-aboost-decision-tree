%bΪ������bint�ع�ϵ����������ƣ�rΪ�в
%rintΪ�������䣬stats���ڻع�ģ�ͼ���
%stats=[r2(���ϵ����r2Խ�ӽ�1˵��������ϳ̶�Խ��),F(Fͳ�������ڼ���ģ����
%��ͨ�����顣ͨ����F�ֲ������F>F�ֲ����ж�Ӧ��ֵ����ͨ������),p(��Լ���
%ԭ����(��:����ϵ��Ϊ��)�����ĸ��ʡ���p<0.05(alphaĬ��ֵ),˵�������Ŀ�����
%��С������5%�������0,p<alphaʱ�ܾ�H0,���ܾ�����裬��Ϊ�ع鷽����������һ
%��������ϵ����Ϊ��ع�ģ�ͳ���),�в��]
%����ع�ģ�ͣ����ϵ��r^2=stats(1,1)Խ�ӽ�1�ع鷽��Խ����
%F=stats(1,2)ֵԽ��ع鷽��Խ����
%p=stats(1,3)<0.05ʱ�ع�ģ�ͳ���
%stats(1,4)�в�ķ��ԽСԽ��
function [RMSEP,R,R2,RPD] = mlr(xtrain,ytrain,xtest,ytest)
pz=[xtrain,ytrain];
[row,col]=size(pz);
x=pz(:,1:col-1);
y=pz(:,end);
[m,n]=size(x);
X=[ones(m,1),x];
[b,bint,r,rint,stats]=regress(y,X);
% stats
% disp('�ع�ģ�ͼ���:');
% if (stats(1,3)<0.05)&(stats(1,1)>0.6)
%    disp('�ع鷽������,ģ�ͳ���');
% end
test=[xtest,ytest];
[row1,col1]=size(test);
aa1=test(:,1:col1-1);%�Ա���	
aa1 = aa1';
xish = b(2:col1)';
bb1=xish; %�ع�ϵ��
ch0 = b(1);%������
c=repmat(ch0,1,row1); %������
s=bb1*aa1+c;
yy=s';
d=test(:,end);%ʵ��ֵ
m=sum((d-yy).^2);
RMSEP=sqrt(m/row1);%���������
R=corrcoef(d,yy);
R=R(1,2);
R2=1-m/sum((d-mean(ytest)).^2);
RPD=1/sqrt(1-R2);
mmin=min(ytest)-1;
mmax=max(ytest)+1;
% figure
% plot(mmin:mmax,mmin:mmax,d,yy,'o');
% xlabel('ʵ��ֵ/��Brix','FontSize',10);%x��
% ylabel('Ԥ��ֵ/��Brix','FontSize',10); %y��
end

