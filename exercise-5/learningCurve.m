function [error_train,error_val]=learningCurve(X,y,Xval,yval,lambda)
%% 函数功能：计算不同训练样本下的优化结果在训练集和验证集上的误差曲线
m = size(X,1);
error_train = zeros(m,1);
error_val = zeros(m,1);
% 每次带入不同数目的训练样本数量，计算在所带入的训练集上的代价，以及在验证集上的代价
for i = 1:m
    theta = trainLinearReg(X(1:i,:),y(1:i),lambda);
    [error_train(i),~] = linearRegCostFunction(X(1:i , :),y(1:i),theta,lambda);
    [error_val(i),~] = linearRegCostFunction(Xval,yval,theta,lambda);
end

end
