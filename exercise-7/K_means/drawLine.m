function drawLine(p1, p2, varargin)
%% 用短直线连接上一步的中心点与当前中心点
plot([p1(1) p2(1)], [p1(2) p2(2)], varargin{:});

end