function [centroids] = kMeansInitCentroids(X,K)
%% 函数功能：从X中随机取K个数据样本，用它们的特征值作为K个随机中心
randIndex = randperm(size(X,1));
centroids  = X(randIndex(1:K),:);

end
