%% Machine Learning  Online Class === Exercise 7:K-Means Clustering and Principle Component Analysis 
close all;clc

%% Optional (ungraded) Exercise: PCA for Visualization--1
A = double(imread('lena.png'));
A = A / 255;
[m,n,l] = size(A);
X = reshape(A,m*n, l);
K = 16; 
max_iters = 10;
init_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, init_centroids, max_iters);

palette = hsv(K);
colors = palette(idx, :);

figure(1);
scatter3(X(:, 1), X(:, 2), X(:,3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
hold off;

%% Optional (ungraded) Exercise: PCA for Visualization--2
[X_norm, mu, sigma] = featureNormalize(X);

[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

figure(2);
plotDataPoints(Z,idx, K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
