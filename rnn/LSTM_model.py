import torch
import torch.nn as nn

class LSTM:
    def __init__(self, Wx, Wh, b): #wx와 wh는 f, g, i, o의 가중치들이 오른쪽으로 쌓여있음
        self.params = [Wx, Wh, b]
        self.grad = [torch.zeros_like(Wx), torch.zeros_like(Wh), torch.zeros_like(b)]
        self.cache = None  #cache는 순전파때 중간결과 보관 후 역전파 계산에 사용
    
    def forward(self, x, hpre, cpre):  #이전시간 은닉, cell 상태
        Wx, Wh, b = self.params
        N, H =hpre.shape

        A = x.mul(Wh) + hpre.mul(Wh) + b

        f = A[:, :H]
        g = A[:, H:H*2]
        i = A[:, H*2:H*3]
        o = A[:, H*3:]

        f = nn.Sigmoid(f)
        g = nn.Tanh(g)
        i = nn.Sigmoid(i)
        o = nn.Sigmoid(o)

        c_next = f * cpre + g * i
        h_next = o * nn.Tanh(c_next)

        self.cache = (x, hpre, cpre, i, f, g, o, c_next)
        return h_next, c_next

#############################################################################################
#교재 LSTM model code

#파라미터 초기화
def get_lstm_params(vsize, nhi):
    nin = nout = vsize

    def normal(shape):
        return torch.randn(size=shape)

    def pra():
        return (normal((nin, nhi)), normal((nhi, nhi)), torch.zeros(nhi))
    W_xi, W_hi, b_i = pra()  
    W_xf, W_hf, b_f = pra()  
    W_xo, W_ho, b_o = pra()  
    W_xc, W_hc, b_c = pra()  

    W_hq = pra((nhi, nout))
    b_q = np.zeros(num_outputs)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

    def init_lstm_state(bsize, nhi):
        return (torch.zeros(shape=(bsize, nhi)),
            torch.zeros(shape=(bsize, nhi)))

    def lstm(inputs, state, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = nn.Sigmoid(np.dot(X, W_xi) + torch.mm(H, W_hi) + b_i)
        F = nn.Sigmoid(np.dot(X, W_xf) + torch.mm(H, W_hf) + b_f)
        O = nn.Sigmoid(np.dot(X, W_xo) + torch.mm(H, W_ho) + b_o)
        C_tilda = nn.Tanh(np.dot(X, W_xc) + torch.mm(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * nn.Tanh(C)
        Y = torhc.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)   #???? torch. cat(outputs)??

