function P = predict_nn(X,Theta1,Theta2)
%% 函数说明：根据隐藏层的参数Thtea1和输出层的参数Theta2，对输入数据X进行预测，获得X在该Neural Networks上的预测结果
% m：数据样本的个数
% n：每一个数据的维度
[m,~] = size(X);
X = [ones(m,1),X];
% m1：隐藏层的unit个数
% n1：隐藏层的每一个unit的θ的维数，大小与输入层数据的维度大小相同
% [m1,n1] = size(Theta1);
% m1：输出层的unit个数
% n1：输出层的每一个unit的θ的维数，大小与隐藏层的输出个数相同
% [m2,n2] = size(Theta2);

% 隐藏层的输入Z1，增加一个值为1的unit
% Z1 的尺寸为m1*m，表示某一个隐藏层unit第一组参数Thteta1下对每一个数据的预测
Z1 = Theta1 * X.' ;
Z1 = [ones(1,m) ; Z1];
% 隐藏层的输出 A1
A1 = sigmoid(Z1);

% 输出层的输入Z2
Z2 = Theta2 * A1;
Z2 = Z2.';

% 输出层的输出，其实没有这个sigmoid函数也行，因为sigmoid是单调增加的函数
A2 = sigmoid(Z2);

[~,P] = max(A2 ,[],2);
end
