function [figurePane , display_array] = displayData(X,image_width)
%% 函数说明：把输入的图像数据X进行重新排列，显示在一个面板figurePane中，面板中有多个小imge用来显示每一行数据
% image_width：每一个小imgae的宽度
% 获取X的具体尺寸
[m,n] =size(X);

%% 设置每一个image的尺寸
% 如果没有设置image_width，就默认为是数据维度的开平方，并四舍五入
if ~exist('image_width','var') || isempty(image_width)
    image_width = round(sqrt(n));
end
% 每一个image的高度
image_height = n / image_width;

%% 设置figurePane（figure）的参数
% 设置面板figurePane图片的色彩为灰度图
colormap(gray);
% 设置面板figurePane中image的行数和列数
figure_rows = floor(sqrt(m));   % floor ---- 向负无穷取整，取小于等于它的最大整数
figure_cols = ceil(m / figure_rows);    % ceil ---- 向正无穷取整，取大于等于它的最小整数

%% 设置面板figurePane对应的数组，用来保存X中的象素值
% 每一个image之间的间距
pad = 1;
% 初始值都是-1，显示为黑色
display_array = -ones( pad + (image_width+pad) * figure_rows, ...
                                     pad+(image_height +pad) * figure_cols );
                                 
%% 把X中的每一个象素值，复制进display_array的对应位置
current_image=1;
for row =1 : figure_rows
    for col =1:figure_cols
        % 判断current_image的大小，大于m就表明遍历结束了
        if current_image > m
            break;
        end
        % 找到每一行的最大值，用来把这一行的数据归一化到[-1,1]之间
        max_val = max( max( X(current_image,:) ) );
        % 按照数据块进行重新放置数据，使用 reshape函数进行位置重排
        % 一个图像数据在重排成一行的时候，也是用reshape方法进行的，再使用reshape方法来恢复
        display_array(pad + (row-1)*(image_height+pad)+(1:image_height) , pad + (col-1)*(image_width+pad)+(1:image_width))=...
            reshape(X(current_image,:),image_height,image_width) / max_val;
       current_image=current_image+1;
    end
    if current_image > m
            break;
    end
end
% 显示图像，并且把色彩（这里是灰度）范围设置为[-1,1]
figurePane = imagesc(display_array,[-1 1]);
title('Random handwritten digits');
axis image off
drawnow;
end
