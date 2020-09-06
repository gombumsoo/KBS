from matplotlib import pyplot as plt
from d2l import torch as d2l
import torch
import torch.nn as nn

T = 1000  # Generate a total of 1000 points
time = torch.arange(0, T, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x])
plt.plot(time, x)
plt.show()

# tau =4
# features = torch.zeros((T-tau,tau))
# print(features.size())
# for i in range(tau):
#     features[:,i]=x[i: T-tau+i]
# labels = d2l.reshape(x[tau:],(-1,1))

# batch_size, n_train =16, 600
# train_iter = d2l.load_array((features[:n_train],labels[:n_train]),batch_size,is_train=True)