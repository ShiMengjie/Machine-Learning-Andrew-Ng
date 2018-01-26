function [centroids] = kMeansInitCentroids(X,K)
%% 生成K个随机中心
randIndex = randperm(size(X,1));
centroids  = X(randIndex(1:K),:);

end
