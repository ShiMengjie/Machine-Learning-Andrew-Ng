function plotData( X, Y)
%% 函数功能：绘制数据中的正负类别
            
%find(x=m)：找到x中值为1的下标向量
pos = find(Y ==1);
neg = find(Y ==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
hold on 
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','yellow','MarkerSize',7);

end
