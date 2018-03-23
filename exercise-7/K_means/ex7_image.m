%% Machine Learning  Online Class === Exercise 7 |K-Means Clustering and Principle Component Analysis 
close all;clc
% 使用K-means实现图像的压缩

% ============== ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ================
% ============== 向为数字图像处理作出巨大贡献的Lena女士，表义崇高的敬意！！！ ==============
% ============== ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ================

%% 1.读取图像，转换成数据数组
X = double(imread('lena.png'));
% X = double(imread('sakimichan.jpg'));
X = X ./ 255; % 归一化到[0,1]
% X是512*512*3的数组
[m,n,l] = size(X);
% 把每一个象素点看作是一个样本，每个样本的维度是l=3，每种颜色看作是一维
X  = reshape(X,m*n,l);
% 把所有样本聚类到50个簇，就是把所有象素都聚类到了50种颜色，用这16种颜色来重构图像，而不是用255色图
K = 50; 
max_iters = 20;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%% 2. 图像压缩：数据压缩 --> 重构图像
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
