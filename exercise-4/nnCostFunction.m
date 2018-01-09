function  [J,grad] = nnCostFunction(X,y,nn_parameter, ...
                                                 input_layer_size, ...
                                                 hidden_layer_size, ...
                                                 label_num, ...
                                                 lambda)
%% 函数功能：根据输入的数据和对应的输出，以及神经网络相应的参数，计算代价函数
if nargin == 6
    %只有6个参数的时候，表示没有正则项，把系数lambda设置为0
    lambda =0;
end
%% 第一部分，计算代价函数
theta1 = reshape(nn_parameter(1:hidden_layer_size * (input_layer_size + 1 )),hidden_layer_size,input_layer_size+1); 
theta2 = reshape(nn_parameter(hidden_layer_size * (input_layer_size + 1 )+1:end),label_num,hidden_layer_size+1);

[m,~] = size(X);
X = [ones(m,1),X];
% 第一层输出
a1 = X;
% 第二层输入和输出
z2 = a1 * theta1.' ;
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) , a2];
% 第三层输入和输出
z3 = a2 * theta2.';
a3 = sigmoid(z3);
% 转换输出向量
yk = zeros(length(y),label_num);
for i =1:size(a3,1)
    yk(i,y(i)) =1;
end

logisf = (-yk) .* log(a3) - (1- yk) .* log(1 - a3);
% 正则项中，没有带入theta1和theta2的偏置部分
J = (1/m) * sum(sum(logisf)) + (lambda) * ( sum(sum(theta1(:,2:end) .^ 2 )) + sum(sum(theta2(:,2:end) .^2)) ) /(2*m);

%% 第二部分，BP算法，按照文档上的写法，但是和我所知的BP算法有些不一样
delta3 = a3 - yk;

delta2 = delta3 * theta2 .* sigmoidGradient([ones(size(z2,1),1),z2]);
delta2 = delta2(:,2:end);
% theta1的梯度
tridelta_1 = 0;
tridelta_1 = tridelta_1 + delta2.' * a1;
nn_parameter1_grad = (1/m) .* tridelta_1 + (lambda/m) *[zeros(size(theta1,1),1),theta1(:,2:end)]  ;
% theta2的梯度
tridelta_2 = 0;
tridelta_2 = tridelta_2 + delta3.' * a2;
nn_parameter2_grad = (1/m) .* tridelta_2 +(lambda/m) * [zeros(size(theta2,1),1),theta2(:,2:end)] ;

grad = [nn_parameter1_grad(:);nn_parameter2_grad(:)];

end