#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
batch_size = 128
#在训练和测试过程中，每次处理的样本数量为 128 个
learning_rate = 0.01
num_epochs = 100
#循环次数
 
transform = transforms.Compose([
    transforms.ToTensor()
])
#它的作用是将输入的图像数据转换为 PyTorch 的张量格式。这种转换使得数据可以被 PyTorch 的神经网络模型处理
 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#train=True 表示加载训练集 transform=transform 应用前面定义的数据预处理转换（这里是将图像转换为张量）到每个数据样本。
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
 #shuffle=True 表示在每个 epoch 开始时随机打乱训练数据的顺序，这有助于提高模型的泛化能力，避免模型过度依赖数据的顺序。
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
#构建一个神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        #将经过卷积和池化后的特征图展平为一维向量。通过两个全连接层，最后返回输出。
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
model = Network().to(device)
 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#均方误差函数 和adam优化器
 #训练模型
def train():
    model.train()
    for epoch in range(num_epochs):
        #这是一个外层循环，用于遍历指定的训练轮数
        running_loss = 0.0
        correct = 0
        total = 0
        #初始化变量用于跟踪当前轮次的累计损失、正确预测的样本数和总样本数。
        for i, data in enumerate(trainloader, 0):
            #内层循环 遍历训练数据加载器（trainloader）生成的批次数据，enumerate函数用于获取批次的数据。
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
 #从数据批次中提取输入图像和对应的标签。然后将它们移动到指定的设备上，以便进行高效的计算。
            optimizer.zero_grad()
 #梯度清零
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
 
            _, predicted = torch.max(outputs.data, 1)
    #找到模型输出中每个样本的预测类别。torch.max函数返回最大值和对应的索引，这里只关心索引（预测的类别）。
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 #更新总样本数和正确预测的的数量
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
 #计算当前轮次的准确率，并打印出当前轮次的信息，包括轮次编号、平均损失和准确率。
#结果评估
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            #提取图像数据和对应的标签，并将它们移动到与模型相同的设备上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            #找到模型输出中每个样本的预测类别。
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
 
if __name__ == "__main__":
    train()
    test()
    
#绘制准确率和损失曲线    
plt.figure(figsize=(10, 5))
#创建一个图形的窗口 10*5
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), accuracies, label='Training Accuracy', color='blue')
#绘制准确率
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
#设立第一个子图的横纵坐标
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss', color='red')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#同理上面
plt.show() 

