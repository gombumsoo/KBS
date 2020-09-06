import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = './MNIST'
batch_size = 256
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_data = dsets.MNIST(root=root, train=True, transform=transform, download=True)
test_data = dsets.MNIST(root=root, train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

#학습파라미터
learning_rate = 0.01
training_epochs = 10
total_batch = len(train_data)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2))

  
        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(120, 84)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(84, 10)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

model = CNN().to(device)

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
    print('Epoch: {:>4}] loss = {:>.9} acc:{:.4f}%'.format(epoch + 1, 
    torch.true_divide(trloss,a),torch.true_divide(tracc,a)))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_loss)
plt.show()
plt.plot(train_acc)
plt.legend()
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