function centroids = computeCentroids(X, idx, K)
%% 函数功能：更新每个簇的中心点，坐标为该簇所有样本坐标和的均值
centroids = zeros(K,size(X,2));

for i=1:K
    index = find(idx == i);
    centroids(i,:) = sum(X(index,:)) / size(X(index),1); 
end

end
