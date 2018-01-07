function init_theta = randInitializeWeights(L_in,L_out)
%% 函数功能说明：初始化层与层之间的权重，输入：L_in，输出：L_out
epsilon_init = 0.12;
init_theta = rand(L_out,L_in+1) * 2* epsilon_init - epsilon_init;

end
