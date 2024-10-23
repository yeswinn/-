#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
#将数据集随机划分为训练集和测试集
from sklearn.preprocessing import StandardScaler
#StandardScaler 用于对数据进行标准化处理 便于算法的收敛
from sklearn.impute import SimpleImputer
#SimpleImputer 可以用来处理这些缺失值。

# 加载并打印数据集
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#这个作图工具我还不是很会，我只能把他做出一个表格的形式
data = pd.read_csv('housing.csv')
data.head()


# In[45]:


print('数据维度',data.shape)


# In[ ]:


# 处理缺失值
imputer = SimpleImputer(strategy='mean')
#用每列的的平均值来代替缺失值
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
#将这个数组转换为 pandas 的 DataFrame，并使用原本数据的列名来命名新数据框的列。

# 对特征进行归一化处理
scaler = StandardScaler()
#创建了一个 StandardScaler 对象，用于对数据进行标准化处理
features = data_imputed.drop('MEDV', axis=1)
#就是我要将我的y值从features中移出来
features_scaled = scaler.fit_transform(features)
#将移除了y值的feature的数据标准化
target = data_imputed['MEDV']
#从 data_imputed 中提取目标变量 MEDV，存储在 target 变量中


#总的来说features_scaled现在就是经过了标准化的x的值，target是我现在的y值

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
#random_state=42 设置随机种子，以确保每次运行代码时得到的划分结果是一致的。


# In[47]:


import torch
import torch.nn as nn
#torch.nn模块提供了构建神经网络的各种组件，包括层、损失函数、激活函数等。通过导入这个模块，你可以方便地定义和构建各种类型的神经网络结构。
import torch.optim as optim
#优化器

# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #这里定义了一个名为Net的类，它继承自torch.nn.Module。在__init__方法中，初始化了三个全连接层（nn.Linear）
        self.fc1 = nn.Linear(3, 32)  # 13个输入特征
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)   # 一个输出特征
#全连接层
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        #前向传播
        return x

# 实例化模型
model = Net()


# In[48]:


# 转换数据为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#将训练集特征X_train转换为torch.float32类型的张量。
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
#将训练集目标变量y_train转换为张量，并通过.view(-1, 1)将其形状调整为列向量形式。
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
#model.parameters()指定了需要优化的模型参数，即神经网络中的权重和偏置

# 训练模型
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    #将优化器中的梯度清零
    outputs = model(X_train_tensor)
    #预测值
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        #每十次打印一次


# In[49]:


# 测试模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

