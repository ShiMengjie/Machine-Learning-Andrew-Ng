function out = mapFeature( X1,X2,power )
%% 函数功能：把特征X1、X2映射成高阶维度的多项式，增加数据的维度
% power是映射到的多项式的最高次数，项数从0次到power次
[m,~]=size(X1);
n=(1+power+1) * (power+1)/2;
out = ones(m,n);
k=2;
for i =1:power
    for j = 0:i
        out(:,k) = (X1.^(i-j)) .* (X2.^(j)); 
        k=k+1;
    end
end
end
