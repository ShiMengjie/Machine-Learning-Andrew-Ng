function plotData( X, Y)
%% 函数功能：绘制数据中的正负类别
%{
%这个写法的问题在于，当使用legend来标注图形示例时，会出错
figure;
for i =1:length(Y)
    if(Y(i) == 1)
        plot(X(i,1),X(i,2),'k+','LineWidth',2,'MarkerSize',7);
        hold on 
    end
    if(Y(i)==0)
        plot(X(i,1),X(i,2),'ko','MarkerFaceColor','yellow','MarkerSize',7);
        hold on
    end
end
%}
            
%find(x=m)：找到x中值为1的下标向量
pos = find(Y ==1);
neg = find(Y ==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
hold on 
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','yellow','MarkerSize',7);

end

