function gd = sigmoidGradient(z)
%% 函数说明：计算在X处的sigmoid函数的导数
gd = sigmoid(z) .* (1 - sigmoid(z));

end
