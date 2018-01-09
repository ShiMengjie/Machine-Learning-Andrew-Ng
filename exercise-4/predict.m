function p = predict(theta1, theta2, X)
m = size(X, 1);

a1 = X;

a2 =sigmoid([ones(m,1),a1] * theta1.');

a3 = sigmoid([ones(m,1),a2] * theta2.');

[~, p] = max(a3, [], 2);

end
