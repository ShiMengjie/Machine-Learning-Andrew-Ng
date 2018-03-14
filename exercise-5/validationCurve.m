function [lambda_vec,error_train,error_val]= validationCurve(X,y,X_val,y_val)
%% 函数功能：选择不同lambda值，求出在训练集所有数据和验证集所有数据上的误差
% lambda序列
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
error_train = zeros(length(lambda_vec),1);
error_val = zeros(length(lambda_vec),1);
for i =1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainLinearReg(X,y,lambda);
    error_train(i) = linearRegCostFunction(X,y,theta,lambda);
    error_val(i) = linearRegCostFunction(X_val,y_val,theta,lambda);
end

end
