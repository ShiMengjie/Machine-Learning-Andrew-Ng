function vocabList = getVocabList()
%% 函数功能：读取固定的文字列表（词袋），并返回这些单词
fid = fopen('vocab.txt');
% 列表中一共1899个单词
n = 1899;  

vocabList = cell(n, 1);
for i = 1:n
    % 先读前面的数字序列但不保存
    fscanf(fid, '%d', 1);
    % 读取并保存一个字符串-单词
    vocabList{i} = fscanf(fid, '%s', 1);
end
fclose(fid);
end