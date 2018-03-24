function [J,grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%% 函数功能：计算代价函数和梯度

X = reshape(params(1:num_movies*num_features), num_movies, num_features);

Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

J = (1/2) * sum(sum((X*Theta.' .* R - Y .* R) .^ 2))+ (lambda / 2) * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));

X_grad = (X*Theta.' .* R - Y .* R) * Theta + lambda * X;

Theta_grad = (X' * (X*Theta.' .* R  - Y .* R)).' + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end
