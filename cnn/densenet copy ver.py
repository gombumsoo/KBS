import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import torchsummary
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = './MNIST'
batch_size = 64
transform = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=0, std=1),
                                #transforms.ToPILImage()
])
train_total_data = dsets.MNIST(root=root, train=True, transform=transform, download=True)
test_total_data = dsets.MNIST(root=root, train=False, transform=transform, download=True)
train_data, r1 = torch.utils.data.random_split(train_total_data, [10000, 50000])
test_data, r2 = torch.utils.data.random_split(test_total_data, [1500, 8500])

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# validation_ratio = 0.1
# random_seed = 10
# initial_lr = 0.1
# num_epoch = 300
learning_rate = 0.1
training_epochs = 5

class bn_relu_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(bn_relu_conv, self).__init__()
        self.batch_norm = nn.BatchNorm2d(nin)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.batch_norm(x)
        out = self.relu(out)
        out = self.conv(out)

        return out

class bottleneck_layer(nn.Sequential):
    def __init__(self, nin, growth_rate, drop_rate=0.2):    
        super(bottleneck_layer, self).__init__()
        
        self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=growth_rate*4, kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('conv_3x3', bn_relu_conv(nin=growth_rate*4, nout=growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.drop_rate = drop_rate
      
    def forward(self, x):
        bottleneck_output = super(bottleneck_layer, self).forward(x)
        if self.drop_rate > 0:
            bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
            
        bottleneck_output = torch.cat((x, bottleneck_output), 1)
        
        return bottleneck_output

class Transition_layer(nn.Sequential):
    def __init__(self, nin, theta=0.5):    
        super(Transition_layer, self).__init__()
      
        self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

class DenseBlock(nn.Sequential):
    def __init__(self, nin, num_bottleneck_layers, growth_rate, drop_rate=0.2):
        super(DenseBlock, self).__init__()
                        
        for i in range(num_bottleneck_layers):
            nin_bottleneck_layer = nin + growth_rate * i
            self.add_module('bottleneck_layer_%d' % i, bottleneck_layer(nin=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10):
        super(DenseNet, self).__init__()
        
        assert (num_layers - 4) % 6 == 0
        
        # (num_layers-4)//6 
        num_bottleneck_layers = (num_layers - 4) // 6
        
        # 32 x 32 x 1 --> 32 x 32 x (growth_rate*2)
        self.dense_init = nn.Conv2d(1, growth_rate*2, kernel_size=3, stride=1, padding=1, bias=True)
                
        # 32 x 32 x (growth_rate*2) --> 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]
        self.dense_block_1 = DenseBlock(nin=growth_rate*2, num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)] --> 16 x 16 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_1 = (growth_rate*2) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_1 = Transition_layer(nin=nin_transition_layer_1, theta=theta)
        
        # 16 x 16 x nin_transition_layer_1*theta --> 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_2 = DenseBlock(nin=int(nin_transition_layer_1*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)] --> 8 x 8 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_2 = int(nin_transition_layer_1*theta) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_2 = Transition_layer(nin=nin_transition_layer_2, theta=theta)
        
        # 8 x 8 x nin_transition_layer_2*theta --> 8 x 8 x [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_3 = DenseBlock(nin=int(nin_transition_layer_2*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)
        
        nin_fc_layer = int(nin_transition_layer_2*theta) + (growth_rate * num_bottleneck_layers) 
        
        # [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)] --> num_classes
        self.fc_layer = nn.Linear(nin_fc_layer, num_classes)
        
    def forward(self, x):
        dense_init_output = self.dense_init(x)
        
        dense_block_1_output = self.dense_block_1(dense_init_output)
        transition_layer_1_output = self.transition_layer_1(dense_block_1_output)
        
        dense_block_2_output = self.dense_block_2(transition_layer_1_output)
        transition_layer_2_output = self.transition_layer_2(dense_block_2_output)
        
        dense_block_3_output = self.dense_block_3(transition_layer_2_output)
        
        global_avg_pool_output = F.adaptive_avg_pool2d(dense_block_3_output, (1, 1))                
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)

        output = self.fc_layer(global_avg_pool_output_flat)
        
        return output

def DenseNetBC_100_12():
    return DenseNet(growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10)

def DenseNetBC_250_24():
    return DenseNet(growth_rate=24, num_layers=250, theta=0.5, drop_rate=0.2, num_classes=10)

def DenseNetBC_190_40():
    return DenseNet(growth_rate=40, num_layers=190, theta=0.5, drop_rate=0.2, num_classes=10)

model = DenseNetBC_100_12()
model.to(device)

# torchsummary.summary(net, (1, 28, 28))
criterion = nn.CrossEntropyLoss().to(device)  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
a=torch.floor_divide(len(train_loader.dataset),batch_size)



train_loss=[]
train_acc=[]
for epoch in range(training_epochs):
    trloss=0
    tracc=0
    for X, Y in train_loader:
        X = Variable(X).cuda()
        Y = Variable(Y).cuda()

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        trloss+=cost
        predict=hypothesis.max(1)[1]
        acc=torch.true_divide((predict.eq(Y).sum())*100, batch_size)
        tracc += acc
    train_loss.append(torch.true_divide(trloss,a))
    train_acc.append(torch.true_divide(tracc,a))
    print('Epoch: {}] loss = {:.4f} acc:{:.4f}%'.format(epoch + 1, 
    torch.true_divide(trloss,a),torch.true_divide(tracc,a)))


import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.title('Loss')
plt.show()

plt.plot(train_acc)
plt.title('Accuarcy')
plt.show()


correct =0
total = 0
with torch.no_grad():
  for data,target in test_loader:
    data=Variable(data).cuda()
    target=Variable(target).cuda()
    outputs=model(data)
    _, predicted=torch.max(outputs.data,1)
    total+=target.size(0)
    correct+=(predicted==target).sum().item()
print('accuracy:%d %%'%(100*correct/total))   