import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

import torchvision 
import matplotlib.pyplot as plt 

########## 数据下载与展示 ##########
# 下载官方MNIST数据集并读取数据集
data_train = torchvision.datasets.MNIST(root='./data', train=True,
                                       transform=torchvision.transforms.ToTensor(), download=True)
data_test = torchvision.datasets.MNIST(root='./data', train=False,
                                      transform=torchvision.transforms.ToTensor(),download=True)

X_train = torch.unsqueeze(data_train.data, 1) # shape = [60000, 1, 28, 28]
y_train = data_train.targets # shape = [60000, 1]
X_test = torch.unsqueeze(data_test.data, 1)
y_test = data_test.targets

print('Sum of train_data(Official): ', len(data_train))
print('Sum of test_data(Official): ', len(data_test))
print('Image shape: ', data_train.data.shape[1:])

############# 训练函数 #############
def train(net, train_features, train_labels, val_features, val_labels, num_epochs, 
          batch_size, optimizer, loss_function, def_init_weight=None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")): 
    """PARAMS:
    net: 需要进行训练的网络结构
    train_features: 训练features数据
    train_labels: 训练targets数据
    val_features: 验证features数据
    val_labels: 验证targets数据
    num_epochs: 需要训练的轮次
    batch_size: batch_size
    optimizer: 定义的优化器(torch.optim)
    loss_function: 自定义的损失函数(torch.nn)
    def_init_weight: 模型参数初始化(默认为None, torch.nn.init)
                        def init_weight(m):
                            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                            nn.init.xavier_uniform_(m.weight)
    device: 自动搜索设备进行运行
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    
    
    if def_init_weight != None:
        net.apply(def_init_weight)
        
    
    train_ls, val_ls = [], [] # 存储train_loss,test_loss
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True) 
    
    net = net.to(device)
    net.train()
    
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y, net = X.to(device), y.to(device), net.to(device)
            output  = net(X.to(torch.float32))
            loss = loss_function(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        # 得到每个epoch的 loss 和 accuracy
        
        with torch.no_grad():
            train_ls.append(celoss_acc(0,net, train_features, train_labels, device=device, loss_func = loss_function))
            # train_ls.append(loss_function(net(train_features.to(device).to(torch.float32)), train_labels))
            if val_labels is not None:
                # val_ls.append(loss_function(net(val_features.to(device).to(torch.float32)), val_labels))
                val_ls.append(celoss_acc(1,net, val_features, val_labels, device=device, loss_func = loss_function))
    
    return train_ls, val_ls

def celoss_acc(flag,net,x,y, device, loss_func):
    if flag == 1: ### valid 数据集
        net.eval()
    x,y, net= x.to(device),y.to(device), net.to(device)
    output = net(x.to(torch.float32))
    result = torch.max(output,1)[1].view(y.size())
    corrects = (result.data == y.data).sum().item()
    accuracy = corrects*100.0/len(y)  #### 5 是 batch_size
    loss = loss_func(output,y)
    net.train()
    
    return (loss.data.item(),accuracy)

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


# 模型训练定义的一些参数
Net_model = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, 5, padding=2, stride=1), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, 5, padding=0, stride=1), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), 
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), 
    nn.Sigmoid(),
    nn.Linear(120, 84), 
    nn.Sigmoid(),
    nn.Linear(84, 10))

# 权重初始化方法
def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

# 超参数
para_num_epoches = 100
para_batch_size = 256
para_optimizer = torch.optim.SGD(Net_model.parameters(), lr = 0.9)
para_loss_func = nn.CrossEntropyLoss()


net =  Net_model ### 实例化Net模型

# 记录时间
time_start = time.time()

train_ls, valid_ls = train(net=net, train_features=X_train, train_labels=y_train, val_features=X_test, val_labels=y_test, \
                            num_epochs=para_num_epoches, batch_size=para_batch_size, optimizer=para_optimizer, \
                            loss_function=para_loss_func, def_init_weight=init_weight)

time_end = time.time()

# 保存train loss和 val loss
np.savetxt('./loss_figures/训练集loss.txt', train_ls)
np.savetxt('./loss_figures/验证集loss.txt', valid_ls)


# 绘制训练集和验证集loss趋势曲线并保存
if not os.path.exists('./loss_figures'):
    os.makedirs('./loss_figures')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决title显示中文字问题
plt.rcParams['axes.unicode_minus'] = False

# 保存训练的效果图
epoches = np.arange(1, para_num_epoches+1)
plt.figure()
plt.plot(epoches, np.array(train_ls)[:, 0], color='b', linestyle='-.', marker='*', lw=2, ms=5, label='train_loss')
plt.plot(epoches, np.array(valid_ls)[:, 0], color='r', linestyle='-.', marker='*', lw=2, ms=5, label='valid_loss')
plt.legend() # 显示图例
plt.xlabel('epoches')
plt.ylabel('loss')
plt.title('训练集与验证集的loss趋势图')
plt.savefig('./loss_figures/训练集与验证集的loss趋势图.png')
# plt.show()

# 保存模型
if not os.path.exists('./model_save'):
    os.makedirs('./model_save')	
torch.save(net, './model_save/Model_#MNIST.pt')       

# 记录train消耗时间
time_train = time_end - time_start

print(' train_time:%.2f seconds\n'%time_train,\
    'train_loss:%.6f'%train_ls[-1][0],'train_acc:%.4f\n'%train_ls[-1][1],\
        'valid loss:%.6f'%valid_ls[-1][0],'valid_acc:%.4f'%valid_ls[-1][1])










