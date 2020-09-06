import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as F 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = './MNIST'
batch_size = 256
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=0, std=1),
                                #transforms.ToPILImage()
])
train_data = dsets.MNIST(root=root, train=True, transform=transform, download=True)
test_data = dsets.MNIST(root=root, train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

#학습파라미터
learning_rate = 0.001
training_epochs = 25

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1=nn.Conv2d(1,96,(11,11),stride=4,padding=1)
        self.conv1_pool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2=nn.Conv2d(96,256,(5,5),padding=2)
        self.conv2_pool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(256,384,(3,3),padding=1)
        self.conv4=nn.Conv2d(384,384,(3,3),padding=1)
        self.conv5=nn.Conv2d(384,256,(3,3),padding=1)
        self.conv5_pool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.flat=nn.Flatten()
        self.fc1=nn.Linear(6400,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,10)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.conv1_pool(x)
        x=F.relu(self.conv2(x))
        x=self.conv2_pool(x)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x=self.conv5_pool(x)
        x=self.flat(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=AlexNet()
use_cuda=True
if use_cuda and torch.cuda.is_available():
      model=AlexNet().cuda()    
# summary(model,input_size=(1,224,224))


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


orrect =0
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
