function Z = projectData(X,U,K)
%% 函数功能：把数据X在U上，投影到K维
Z = X * U(:,1:K);

end
