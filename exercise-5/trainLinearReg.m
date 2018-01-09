function [theta] = trainLinearReg(X,y,lambda)
%% 函数说明：训练参数，求出最优解
init_theta = ones(size(X,2),1);
costFunction = @(t)linearRegCostFunction(X,y,t,lambda);
options = optimset('GradObj','on','MaxIter',200);

theta = fmincg(costFunction,init_theta,options);

end
