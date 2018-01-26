%% Machine Learning  Online Class === Exercise 7 |K-Means Clustering and Principle Component Analysis 
% Image Compression
clear all;clc
%% 1.Load image
X = double(imread('lena.png'));
X = X ./ 255; % 归一化到[0,1]
[m,n,l] = size(X);
% 把每一个象素点看作是一个样本，每个样本的维度是l=3
X  = reshape(X,m*n,l);
% 把所有样本聚类到16个簇，就是把所有象素都聚类到了16种颜色，用这16种颜色来重构图像，而不是用255色图
K = 50; 
max_iters = 20;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%% 2. Image Compression
% 找到重构的每一个象素的颜色值
X_recovered = centroids(idx,:);
X_recovered = reshape(X_recovered,m,n,l);
X = reshape(X,m,n,l);

subplot(1, 2, 1);
imagesc(X); 
title('Original');

subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
