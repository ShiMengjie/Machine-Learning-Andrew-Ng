function [model] = svmTrain(X, Y, C, kernelFunction, ...
                            tol, maxIter)
%% 使用SMO的简化版进行SVMDEXUNLIAN
% X 是数据矩阵，X(i,j)表示di个样本的第j维特征
% Y 是X对应的列向量kabel，y={0,1}
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

eta = 0;
L = 0;
H = 0;

%% 生成核函数矩阵
if strcmp(func2str(kernelFunction), 'linearKernel')
    % K(i,j)的值是，第i个向量与第j个向量的内积值
    K = X * X';
elseif contains(func2str(kernelFunction), 'gaussianKernel')
    % This is equivalent to computing the kernel on every pair of examples
    X2 = sum(X.^2, 2);
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
    K = kernelFunction(1, 0) .^ K;
else
    % Pre-compute the Kernel Matrix
    K = zeros(m);
    for i = 1:m
        for j = i:m
             K(i,j) = kernelFunction(X(i,:)', X(j,:)');
             K(j,i) = K(i,j); %the matrix is symmetric
        end
    end
end

%% 开始训练参数
dots = 12;
while passes < maxIter
    
    num_changed_alphas = 0;
    for i = 1:m
        % 见讲义的公式(12、13)，只不过带入的测试数据是第i个向量X(i,:)
        % E(i)是带入X(i,:)，得到的预测与真值的差，E=((wT * X) + b) - y
        E(i) = b + sum (alphas .* Y .* K(:,i)) - Y(i);
        if ((Y(i)*E(i) < - tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0))
            % 选取第二个alpha，要求两个alpha不同
            j = ceil(m * rand());
            while j == i
                j = ceil(m * rand());
            end
            E(j) = b + sum (alphas .* Y .* K(:,j)) - Y(j);
            
            % Save old alphas
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % 计算边界L和H 
            if (Y(i) == Y(j))
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
           
            if (L == H)
                % continue to next i. 
                continue;
            end

            % Compute eta by (14).
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0)
                % continue to next i. 
                continue;
            end
            
            % Compute and clip new value for alpha j using (12) and (15).
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            
            % Clip
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            
            % Check if change in alpha is significant
            if (abs(alphas(j) - alpha_j_old) < tol),
                % continue to next i. 
                % replace anyway
                alphas(j) = alpha_j_old;
                continue;
            end
            
            % Determine value for alpha i using (16). 
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            % Compute b1 and b2 using (17) and (18) respectively. 
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

            % Compute b by (19). 
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

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end

%% 保存输出模型
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = kernelFunction;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';

end
