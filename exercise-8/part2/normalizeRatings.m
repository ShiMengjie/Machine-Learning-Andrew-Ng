function [Ynorm, Ymean] = normalizeRatings(Y, R)
%% 函数功能：求出每个电影的评分均值，并把分数去均值
[m, ~] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    Ymean(i) = mean(Y(i, R(i, :)));
    Ynorm(i, R(i, :)) = Y(i, R(i, :)) - Ymean(i);
end

end
