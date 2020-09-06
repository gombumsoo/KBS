from d2l import torch as d2l
import torch

def corr2d_multi_in(X,K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))  # 동일한 모형의 X, k 묶어줌

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
aa=zip(X,K)

# print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

T = torch.stack((K, K + 1, K + 2), 0)
T.shape
# print(corr2d_multi_in_out(X, T))

# print(corr2d_multi_in(X,K+1 ))

#6.4.3
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)  # Matrix multiplication in the fully-connected layer
    return Y.reshape((c_o, h, w))
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(torch.abs(Y1 - Y2))) < 1e-6



#return sum(d2l.corr2d(x, k) for x, k in zip(X, K)) 어떻게 작동되는지