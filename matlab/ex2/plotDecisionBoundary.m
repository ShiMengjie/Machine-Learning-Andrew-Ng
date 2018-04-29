function plotDecisionBoundary( theta,X,Y )
%% 函数功能：根据theta，数据特征X和数据输出Y，绘制分界线
% X：添加了偏置维度的数据集

plotData(X(:,2:3),Y);
hold on 
%分界线就是X*theta=0.5的直线线
%根据公式计算出，在这个判别结果下，X第一纬度的值对应的第二维度的每一个值
%数据维度不大于3维
if size(X, 2) <= 3
    % 只需要两个点(起始点和终点)就可以绘制出一条直线，对应的是X(:,2)的最小值和最大值
    plot_x = [min(X(:,2)),  max(X(:,2))];
    % sigmod函数的判别线是0.5，对应的theta*X值为0，所以用0减去
    plot_y = (-1 ./ theta(3)) .* (theta(2) .* plot_x + theta(1));
    plot(plot_x, plot_y)
else
    % 数据维度大于3维，绘制等高线
    %u，v是显示的两个维度的取值范围，这里作为网格坐标
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);  
    % 在contour(X,Y,Z)中，要求：
    % length(X) = size(Z,2);
    % length(Y) = size(Z,1);
    % 所以u是纵坐标，v是横坐标
    z = zeros(length(u), length(v));
    % 
    power = 6;
    % 计算在网格每一个点上的值，保存进z
    for i = 1:length(v)
        for j = 1:length(u)
            % z是当前theta和维度映射后的值
            % 需要注意，z(j,i)的值由[X(i), Y(j)]计算出来的
            z(j,i) = mapFeature(v(i), u(j),power) * theta;
        end
    end
    %只需要显示z=0的等高线，指定等高线显示的范围[0,0]
    contour(v, u, z,[0,0],'g-');
end
end

