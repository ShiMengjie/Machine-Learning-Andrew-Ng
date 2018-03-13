function init_theta = randInitializeWeights(input_layer , output_layer)
%% 函数功能：初始化层与层之间的权重，输入：L_in，输出：L_out
epsilon_init = sqrt(6) / sqrt(input_layer + output_layer);
init_theta = rand( output_layer,input_layer+1) * 2* epsilon_init - epsilon_init;

end