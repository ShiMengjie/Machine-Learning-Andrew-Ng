function [C,sigma] = dataset3Params(X, y,Xval,yval,Cs,sigmas)
%% 通过训练集和验证集，找到最合适的C和sigma
% 注意：这不是训练SVM，训练SVM是在svmTrain中进行的
% 带入不同的C和sigma值的组合，找到在验证集上错误率最小的组合
m=length(Cs);
n=length(sigmas);
error = zeros(m,n);

for i=1:m
    C=Cs(i);
    for j = 1:n
        sigma = sigmas(j);
        model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
        p = svmPredict(model,Xval);
        error(i,j) = mean(double(p~=yval))*100;
    end
end

[row,col]=find(error == min(min(error)));
% 如果有多个值，取最小的一个
C= min(Cs(row));
sigma = min(sigmas(col));

end
