import torch
import torch.nn as nn
from torch.nn import functional as F


#가중치 초기화
def get_params(vsize, nhi, device):
    nin = nout = vsize
    W_xh = torch.randn(nin, nhi)
    W_hh = torch.randn(nhi, nhi)
    b_h = torch.zeros(nhi)
    W_hq = torch.randn(nhi, nout)
    b_q = torch.zeros(nout)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

#hidden state 초기화
def init_state(bsize, nhi):
    return(torch.zeros(bsize, nhi))

#hidden cell 구조
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch: 
    """A RNN Model based on scratch implementations."""
    def __init__(self, vsize, nhi, get_params, init_state, forward):
        self.vsize, self.nhi = vsize, nhi
        self.params = get_params(vsize, nhi)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = F.one_hot(X.T.long(), self.vsize).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, bsize):
        return self.init_state(bsize, self.nhi)

nhi=512
model = RNNModelScratch(len(vocab), nhi,  get_params, init_state, rnn)
state = model.begin_state(X.shape[0])
Y, new_state = model(X, state)