function p = predict(theta,X )
%% 函数功能：根据得到的参数，对数据进行预测，测试输出
p = X * theta;
p(p>0.5)=1;
p(p<0.5)=0;

end
