function X_rec = recoverData(X_pro,U,K)
%% 从投影数据X_pro恢复出原始数据

X_rec=X_pro * U(:,1:K).';

end
