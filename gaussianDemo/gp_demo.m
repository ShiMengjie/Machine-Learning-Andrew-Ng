%%  高斯过程回归
close all;clc
% check if running Octave
v = version;
is_octave = v(1) < '5'; 
% 定义输入集x的范围
x_min = 0;
x_step = 0.1;
x_max = 10;
X = (x_min:x_step:x_max).';
%% 定义用到的参数
% 定义核函数以及Kse核函数的带宽τ
tau = 1.0;
kernel = @(x,z) exp(-(x-z).' * (x-z) / (2*tau^2)); % 高级函数句柄
% 噪声满足分布的标准差
sigma = 0.1;

%%
while (true)
    disp('CS 229 Gaussian processes demo');
    disp('------------------------------');
    disp(' ');
    fprintf('1.Set kernel bandwidth (currently tau = %f)', tau);
    fprintf('2.Set noise level (currently sigma = %f)', sigma);
    disp('3.Show samples from a Gaussian process prior.');
    disp('4.Create a random regression problem.');
    disp('5.Run Gaussian process regression.');
    disp('6.Add more training data.');
    disp(' ');
    disp('Q.  Quit');
    disp(' ');
    % 获取输入的字符并全部转成小写
    reply = lower(input('Select a menu option: ', 's'));
    switch (reply)
        % 1.设置τ
        case {'1'}
            tau = max(1e-8, input('Enter desired kernel bandwidth: '));
            kernel = @(x,z) exp(-(x-z)'*(x-z) / (2*tau^2));
        % 2.设置sigma
        case {'2'}
            sigma = max(1e-8, input('Enter desired noise level: '));
        % 3.显示高斯过程先验分布中的样本
        case {'3'}
            % 在每一个x值处不止有一个值，有多个h函数值h(x)，这里设置随机函数个数为3
            num_samples = 3;
            h = zeros(size(X,1), num_samples);
            for i=1:num_samples
                h(:,i) = sample_gp_prior(kernel,X);
            end
            newplot;
            plot(repmat(X,1,num_samples), h, 'LineWidth', 3);
            title(sprintf('Samples from GP with k(x,z) = exp(-||x-z||^2 / (2*tau^2)), tau = %f', tau));
            legend('h1(x)','h2(x)','h3(x)');
        % 4.产生一个随机高斯回归问题--绘制一个有噪声的随机函数
        case {'4'}
            % 生成一个随机函数h
            h = sample_gp_prior(kernel,X);
            y_min = min(h) - 1;
            y_max = max(h) + 1;
            % 取10个数据点，添加随机噪声作为训练数据点
            m_train = 10;
            i_train = floor(rand(m_train,1) * size(X,1) + 1);
            X_train = X(i_train);
            y_train = h(i_train) + sigma * randn(m_train,1);

            newplot;
            plot(X, h, 'k', 'LineWidth', 3);
            hold on;
            plot(X_train, y_train, 'rx', 'LineWidth', 5);
            axis([x_min x_max y_min y_max]);
            title(sprintf('Random regression problem with N(0,sigma^2) noise, sigma = %f', sigma));
            hold off;
        % 5.运行高斯回归过程
        case {'5'}
            X_test = X;% 测试集
            % 计算高斯过程核函数矩阵中的4个核函数矩阵
            K_train_train = compute_kernel_matrix(kernel,X_train,X_train);  
            K_test_train = compute_kernel_matrix(kernel,X_test,X_train);
            K_test_test = compute_kernel_matrix(kernel,X_test,X_test);
            % 计算预测的参数
            G = K_train_train + sigma^2 * eye(size(X_train,1));
            mu_test = K_test_train * (G \ y_train);% 预测的均值函数
            Sigma_test = K_test_test + sigma^2 * eye(size(X_test,1)) - K_test_train * (G \ (K_test_train'));% 预测的方差矩阵
            stdev_test = sqrt(diag(Sigma_test));% 返回主对角线上的元素，开平方得到标准差
            
            newplot;
            hold on;
            if (is_octave)
                errorbar(X_test, mu_test, stdev_test, '~');
            else
                % 取95%置信区间
                % 因为fill函数的用法，需要形成一个闭环
                lower_test = mu_test - 2*stdev_test;
                upper_test = mu_test + 2*stdev_test;
                region_X = [X_test; X_test(end:-1:1,:)];
                region_Y = [lower_test; upper_test(end:-1:1,:)];
                fill(region_X, region_Y, 'g');
            end
            plot(X, h, 'k', 'LineWidth', 3);
            plot(X_train, y_train, 'rx', 'LineWidth', 5);
            % 绘制均值，在均值处概率最高
            plot(X_test, mu_test, 'r', 'LineWidth', 3);
            axis([x_min x_max y_min y_max]);
            title('Gaussian process regression, 95% confidence region');
            hold off;            
        % 6.添加更多的训练数据进行训练
        case {'6'}
            % 每次增加10个训练样本进入训练集
            m_train = 10;
            i_train = floor(rand(m_train,1) * size(X,1) + 1);
            X_train = [X_train; X(i_train)];
            y_train = [y_train; h(i_train) + sigma * randn(m_train,1)];
            
            newplot;
            hold on;
            plot(X, h, 'k', 'LineWidth', 3);
            plot(X_train, y_train, 'rx', 'LineWidth', 5);
            axis([x_min x_max y_min y_max]);
            title('Random regression problem');
            hold off;     
        % Q：返回
        case {'q'}
            break;
    end
end

%% 注意：
% 这里并没有训练的过程：
% 1.使用的是固定的超参数{sigma,sigma,...}
% 2.测试数据就是整个训练集
% 3.通过增加训练样本的数量，获得更加准确的均值和方差矩阵，来拟合已经存在的数据曲线
