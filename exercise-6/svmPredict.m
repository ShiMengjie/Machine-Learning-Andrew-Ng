function pred = svmPredict(model,Xtest)
%% 函数功能：预测SVM在测试数据上的输出
[m,~] = size(Xtest);
pred = zeros(m, 1);

K = model.kernelFunction(model.X,Xtest);
K = model.y .* model.alphas .* K;
p = sum(K,1) + model.b;

pred(p >= 0) =  1;
pred(p < 0) =  0;

end
