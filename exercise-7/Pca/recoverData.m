function X_rec = recoverData(X_pro,U,K)
%% 函数功能：投影数据X_pro从K个主成分中恢复出原始数据

X_rec=X_pro * U(:,1:K).';

end
