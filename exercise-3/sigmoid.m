function [out ] = sigmoid(z)
%% logsitic º¯Êý
out = 1.0 ./ (1.0+exp(-z));
end