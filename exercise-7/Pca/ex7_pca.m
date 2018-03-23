%% Machine Learning  Online Class === Exercise 7 |K-Means Clustering and Principle Component Analysis 
close all;clc

%% 1.读取数据
X = importdata('ex7data1.mat');
figure(1);
plot(X(:,1),X(:,2),'bo','LineWidth',2);
xlabel('X feature 1');
ylabel('X feature2');
axis([min(X(:,1))-0.5 max(X(:,1))+0.5 2 8]);
% axis square:当前坐标系图形设置为方形
% axis equal 将横轴纵轴的定标系数设成相同值 ,即单位长度相同，
axis square;
hold on;

%% 2.PCA过程
% 1.标准化数据
[X_norm,mu,~] = featureNormalize(X);
% 2.对标准化后的数据进行PCA
[U,S] = pca(X_norm);
% 绘制两个主成分(因为原始数据只有2维)
drawLine(mu.', mu.'+S(1,1)*U(:,1), '-k','LineWidth',2);
hold on
drawLine(mu.', mu.'+S(2,2)*U(:,2), '-k','LineWidth',2);
hold off;

%% 3. 投影数据到主向量上，并恢复数据
figure(2);
% 显示标准化后的数据
plot(X_norm(:, 1), X_norm(:, 2), 'bo','LineWidth', 2);
axis([min(X_norm(:,1))-0.5 max(X_norm(:,1))+0.5 min(X_norm(:,2))-0.5 max(X_norm(:,2))+0.5]); 
axis square
hold on
% 原始数据只有2维，投影到1维，Z其实是投影到新的基向量上的系数
K=1;
Z = projectData(X_norm,U,K);
% 从投影数据恢复出原始数据在新的基函数上的投影
X_rec = recoverData(Z,U,K);
plot(X_rec(:,1),X_rec(:,2),'ro','LineWidth', 2);

for i=1:size(X_norm,1)
    drawLine(X_norm(i,:),X_rec(i,:),'--g','LineWidth', 2);
end
hold off;

%% 4. 读取脸部数据
facedata = importdata('ex7faces.mat');
% 显示前100个样本
figure(3);
displayData(facedata(1:100,:));
hold off;
[face_norm,mu,~] = featureNormalize(facedata);
[U,S] = pca(face_norm);
% 绘制前36个特征向量
figure(4);
displayData(U(:,1:36).');
hold off

%% 5.从基函数和系数中使用指定的主成分数恢复原始数据
K=150;
Z_face = projectData(face_norm,U,K);
face_rec = recoverData(Z_face,U,K);
sigma = S * ones(size(S,2),1);
percent = sum(sigma(1:K)) / sum(sigma);
fprintf('The percentage is %f\n',percent);
figure(5);
subplot(1,2,1);
displayData(face_norm(1:K,:));
subplot(1,2,2);
displayData(face_rec(1:K,:));
hold off;
% 脸图像的主要结构还是在的，但是细节部分丢失了
