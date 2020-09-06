from d2l import torch as d2l
import torch
from torch import nn

def corr2d(X,K):
    """Compute 2D cross-correlation."""
    h, w=K.shape   #h는 K의 행, w는 K의 열을 나타냄
    Y = torch.zeros((X.shape[0]- h +1, X.shape[1]-w+1)) 
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))      #reduce_sum 해당차원에서 계산?
    return Y
#ch6.2.1
# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# aa=corr2d(X, K)              

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
#6.2.3
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
Y
print(Y)

#6.2.4
conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
print(Y)
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')

d2l.reshape(conv2d.weight.data, (1, 2))




#질문할거  reduce_sum