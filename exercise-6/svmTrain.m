function [model] = svmTrain(X, Y, C, func, ...
                            tol, maxIter)
%% 函数功能：使用SMO的简化版进行SVM训练
% 关于SMO的公式推导过程，见以下博客和论文：
% http://blog.csdn.net/v_july_v/article/details/7624837
% Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
% X 是数据矩阵，X(i,j)表示第i个样本的第j维特征
% Y 是X对应的列向量label，输入时y={0,1}
% C 是SVM的正则化参数
% tol 参数满足KKT条件的误差
% maxIter 最大迭代运算次数
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end
if ~exist('max_passes', 'var') || isempty(maxIter)
    maxIter = 5;
end

%% 准备数据和参数
m = size(X, 1);
% 在SVM中，label只有{-1,+1}两种，所以要把0值改成-1
Y(Y==0) = -1;
% 在（对偶）优化问题中需要用到的参数
alphas = zeros(m, 1); % 拉格朗日乘子
b = 0; % 偏置
passes = 0; % 迭代次数
E = zeros(m, 1); % 误差矩阵，E(i)表示第i个向量作为测试/训练数据时的误差

%% 生成核函数矩阵
K = func(X,X);

%% 开始训练参数
while passes < maxIter
    num_changed_alphas = 0;
    for i = 1:m
        % E(i)是带入X(i,:)，得到的预测与真值的差，E=((wT * X) + b) - y
        E(i) = b + sum (alphas .* Y .* K(:,i)) - Y(i);
        % Y(i)*E(i) = y((wT * X) + b) - 1
        % 在不满足KKT条件的乘子中找到两个乘子alpha1和alpha2,
        if ((Y(i)*E(i) < - tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0))
            % 选取alpha2，要求两个alpha不同
            j = ceil(m * rand());
            while j == i
                j = ceil(m * rand());
            end
            E(j) = b + sum (alphas .* Y .* K(:,j)) - Y(j);
            
            % 保存当前的值，在后面计算中要用到
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % 根据两个乘子对应的y(i)\y(j)的符号是否相同，来计算边界L和H 
            if (Y(i) == Y(j))
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
            % 如果L=H，表示不需要更新，直接进入下一轮
            if (L == H) 
                continue;
            end

            % eta = K(i,i)+K(j,j)-2*K(i,j)，要求值大于0才能继续计算，否则退出
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0)
                continue;
            end
            
            % 更新alpha2的值,考虑L和H的约束
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
 
            % 更新alpha1的值
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            % 计算b1和b2
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,i)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';
            % 计算b 
            if (0 < alphas(i) && alphas(i) < C)
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C)
                b = b2;
            else
                b = (b1+b2)/2;
            end

            num_changed_alphas = num_changed_alphas + 1;
        end
    end
    
    if (num_changed_alphas == 0)
        passes = passes + 1;
    else
        passes = 0;
    end
end

%% 保存输出模型
% 仅仅保存支持向量对应的alphas
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = func;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';

end
