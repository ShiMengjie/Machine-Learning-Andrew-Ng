function [xplot,yplot,vals]=visualizeBoundary(X, model)
%% 函数功能：绘制非线性分界面
% 把平面看作是一个网格，在每一个网格点上做预测，并绘制每一个点的预测值
xplot = linspace(min(X(:,1)), max(X(:,1)), 100)';
yplot = linspace(min(X(:,2)), max(X(:,2)), 100)';
% 生成网格
% X1每一行都是xplot的复制，一共有length(yplot)行
% X2每一列都是yplot的复制，一共有length(xplot)列
[X1,X2] = meshgrid(xplot, yplot);
% 形成一个10000*2的矩阵，第一列每100行是X1中的1列，第二列每100行是X2中每一列
mat = [X1(:),X2(:)];
% 一次性计算所有网格点的预测值，再重排成需要的格式，不使用循环只使用矩阵运算
vals = reshape(svmPredict(model, mat).',length(yplot),length(xplot));

end