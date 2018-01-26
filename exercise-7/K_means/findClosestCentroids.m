function [idx] = findClosestCentroids(X,init_centroids)
%% step1：找到X每个样本所属的簇
% init_cetroids 的列数和X的列数相等
[m,~] = size(X);
k = size(init_centroids,1);

distance = zeros(m,k);

for i =1:k
    distance(:,i) = sum(bsxfun(@minus,X,init_centroids(i,:)) .^ 2,2) ;
end
[~,I] = min(distance.');
idx = I.';

end
