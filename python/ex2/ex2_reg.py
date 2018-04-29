import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from plotData import plot_data
from costFunction import cost_Function_Reg 
import plotDecisionBoundary as pdb
import predict as predict
import mapFeature as mf



# ============ 1.读取数据
# The first two columns contain the exam scores and the third column contains the label.
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

plot_data(X, y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
# plt.show()

# 2.投影到高维，计算带正则项的代价和梯度

X = mf.map_feature(X[:, 0], X[:, 1],6)
print(X.shape)
# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lmd = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost, grad = cost_Function_Reg (X, y,initial_theta, lmd)

np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})
print('Cost at initial theta (zeros): {: 0.4f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')


# Compute and display cost and gradient with non-zero theta
test_theta = np.ones(X.shape[1])

cost, grad = cost_Function_Reg( X, y,test_theta, lmd)

print('Cost at test theta: {}'.format(cost))
print('Expected cost (approx): 2.13')
print('Gradient at test theta - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.3460\n 0.0851\n 0.1185\n 0.1506\n 0.0159')


# ===================== Part 2: Regularization and Accuracies =====================
# Optional Exercise:
# In this part, you will get to try different values of lambda and
# see how regularization affects the decision boundary
#
# Try the following values of lambda (0, 1, 10, 100).
#
# How does the decision boundary change when you vary lambda? How does
# the training set accuracy vary?
#

# Initializa fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lmd = 0.1

# Optimize
def cost_func(t):
    return cost_Function_Reg(X, y,t, lmd)[0]

def grad_func(t):
    return cost_Function_Reg(X, y,t, lmd)[1]

theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta, maxiter=400, full_output=True, disp=False)

# Plot boundary
print('Plotting decision boundary ...')
pdb.plot_Decision_Boundary( X, y,theta)
plt.title('lambda = {}'.format(lmd))

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
# Compute accuracy on our training set
p = predict.predict( X,theta)

print('Train Accuracy: {:0.4f}'.format(np.mean(y == p) * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

