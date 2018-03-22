function x = emailFeatures(word_indices)
%% 函数功能：把一个文字标记表，转换成特征向量，把有值的位置改为1
n = 1899;
x = zeros(n, 1);

for i=1:length(word_indices)
	x(word_indices(i))=1;
end

end