%% %% Machine Learning Online Class --- Exercise-6:Spam emails
close all;clc
% 添加函数路径
addpath(genpath('./kernel'));
addpath(genpath('./txt'));
addpath(genpath('./methods'));

%% 1.文本预处理和提取特征
% 读取样本邮件的内容
file_contents = readFile('emailSample1.txt'); 
% 把邮件内容转换成文字特征标记
word_indices = processEmail(file_contents);   

fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

feature = emailFeatures(word_indices);
fprintf('Length of feature vector: %d\n', length(feature));
fprintf('Number of non-zero entries: %d\n', sum(feature > 0));

%% 2.导入数据、训练SVM模型和测试模???
% 导入训练集
load('spamTrain.mat');
C=0.1;
model = svmTrain(X,y,C,@linearKernel);
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);
% 导入测试集
load('spamTest.mat');
p = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

%% 3.对w进行降序排列，显示前15个权重最高的单词
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

%% 4.对邮件进行的预测分类
filename = 'spamSample1.txt';
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x = emailFeatures(word_indices).';
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
