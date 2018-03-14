function [theta] = trainLinearReg(X,y,lambda)
%% 函数说明：训练线性回归的参数，返回最优解参数
init_theta = ones(size(X,2),1);
costFunction = @(t)linearRegCostFunction(X,y,t,lambda);
options = optimset('GradObj','on','MaxIter',200);

theta = fmincg(costFunction,init_theta,options);

end
