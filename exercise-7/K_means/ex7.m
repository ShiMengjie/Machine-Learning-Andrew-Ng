%% Machine Learning  Online Class === Exercise 7 K-Means Clustering and Principle Component Analysis
close all;clc

%% 1.读取数据
X = importdata('ex7data2.mat');
figure(1);
plot(X(:,1),X(:,2),'ko','LineWidth',2);
xlabel('Feature 1');
ylabel('Feature 2');
title('Visualizing data');
hold off;

%% 2. 寻找每一个样本所属的“簇”
% 从figure1中，可以看出，数据可以分成3簇
K=3;
% 随机的K个簇中心坐标
init_centroids = [3,3; 6,2; 8,5];
% 找到每个数据最近的中心坐标号：1~K
idx = findClosestCentroids(X,init_centroids);

%% 3. 绘制计算每个簇中心点的过程
% 运行K-means，并绘制中心点的计算过程
maxIter =10;
[~,~] = runkMeans(X,init_centroids,maxIter,true);
