function p = predict(theta,X )
%% 函数功能：根据得到的参数，对数据进行预测，测试输出
p=X*theta;
for i = 1:length(p)
    if(p(i) >= 0.5)
        p(i)=1;
    else
        p(i)=0;
    end
end

end
