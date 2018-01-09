function [error_train,error_val]=learningCurve(X,y,Xval,yval,lambda)
%% 函数说明：在训练集和验证集上求出误差error随着训练样本数目变化的曲线
m = size(X,1);
error_train = zeros(m,1);
error_val = zeros(m,1);

for i = 1:m
    theta = trainLinearReg(X(1:i,:),y(1:i),lambda);
    [error_train(i),~] = linearRegCostFunction(X(1:i,:),y(1:i),theta,lambda);
    [error_val(i),~] = linearRegCostFunction(Xval,yval,theta,lambda);
end

end
