import torch
from torch import nn
import torch.nn.functional as F 

# class BN_block(input_channels,num_channels,drepRate=0.0):
#     def __init__(self):
#         self.BN=nn.BatchNorm2d(input_channels)
#         self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1)
#         self.droprate = dropRate

#     def forward(self, x):
#         out=self.conv1(F.relu(self.BN(x)))
#         if self.droprate>0:
#             out=F.dropout(out, p=self.droprate,training = self.training
#         return torch.cat([x,out],1)

# class BottleneckBlock(nn.Module):
#     def __init__(self, nin,nout,dropRate=0.0):
#         super(BottleneckBlock,self).__init__()
#         inter = nout*4    #ottleneck layer의 conv 1x1 filter chennel 수는 4*growh_rate이다
#         self.BN1=nn.BatchNorm2d(nin)
#         self.conv1=nn.Conv2d(nin,inter,kernel_size=1,stride=1,padding=1,bias=False)
#         self.BN2=nn.BatchNorm2d(inter)
#         self.conv2=nn.Conv2d(inter,nout,kernel_size=3,stride=1,padding=1,bias=False)
#         self.droprate = dropRate
        
#     def forward(self, x):
#         out=self.conv1(F.relu(self.BN1(x)))
#         if self.droprate>0: 
#             out = F.dropout(out,p=self.droprate,inplace=False,training = self.training) 
#         out = self.conv2(self.relu(self.bn2(out))) 
#         if self.droprate>0: 
#             out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
#         return torch.cat([x,out],1)



# # class DenseBlock(nn.Module):
# #     def __init__(self, nin,nout,num_channels):
# #         super(DenseB,self).__init__()
# #         for i in range(num_convs):

# class DenseBlock(nn.Module): 
    # def __init__(self,nb_layers,in_planes,growh_rate,block,dropRate=0.0): 
    #     super(DenseBlock,self).__init__() 
    #     self.layer = self._make_layer(block, in_planes, growh_rate, nb_layers, dropRate) 
    
    # def _make_layer(self,block,in_planes,growh_rate,nb_layers,dropRate): 
    #     layers=[] 
    #     for i in range(nb_layers): 
    #         layers.append(block(in_planes + i*growh_rate ,growh_rate,dropRate)) 
    #     return nn.Sequential(*layers) 
    
    # def forward(self,x): 
    #     return self.layer(x)

