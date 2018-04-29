%% Machine Learning Online Class --- Exercise 8 | Anomaly Detection and Collaborative Filtering
close all;clc
% 使用协同滤波，来进行电影推荐
%% 1.载入数据
load ('ex8_movies.mat');
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1,R(1,:))));
figure(1);
imagesc(Y);
ylabel('Movies');
xlabel('Users');
hold off;

%% 2.计算代价
load ('ex8_movieParams.mat');
% 直选中其中的小部分数据进行计算，验证函数的正确性
num_movies = 5;
num_features = 3;
num_users = 4; 
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

params = [X(:);Theta(:)];
J = cofiCostFunc(params, Y, R, num_users, num_movies, ...
               num_features, 0);
           
fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n'], J);

%% 3.检查梯度
fprintf('\nChecking Gradients (without regularization) ... \n');

checkCostFunction;

%% 4.带入正则项
clc;
lambda = 1.5;
J = cofiCostFunc(params, Y, R, num_users, num_movies, ...
               num_features, lambda);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);

checkCostFunction(lambda);

%% 5.读取电影列表并给一些电影随机评分
movieList = loadMovieList();
my_ratings = zeros(length(movieList), 1);

% 随机对一些电影给评分
my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
index = find(my_ratings > 0);
for i=1:length(index)
     fprintf('Rated %d for %s\n', my_ratings(index(i)), movieList{index(i)});
end

%% 6.进行协同滤波训练和学习
load('ex8_movies.mat');
% 把前面的随机评分添加进Y和R矩阵中
Y = [my_ratings,Y];
R = [(my_ratings ~= 0),R];
% 去均值
[Ynorm, Ymean] = normalizeRatings(Y, R);
% 设置参数
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;
% 设置初始值
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];
% 训练
lambda = 10;
options = optimset('GradObj', 'on', 'MaxIter', 100);
costFun = @(t) cofiCostFunc(t, Y, R, num_users, num_movies,num_features, lambda);
theta = fmincg (costFun ,initial_parameters, options);

X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

%% 7.使用上面训练出的X和Theta，进行推荐
% 计算出预测的评分
p = X * Theta';
my_predictions = p(:,1) + Ymean;
% 选出评分最高的10部电影进行推荐
[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i = 1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', r(i),movieList{j});
end
