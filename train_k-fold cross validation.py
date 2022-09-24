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

print('Sum of train_data(Official): ', len(data_train))
print('Sum of test_data(Official): ', len(data_test))
print('Image shape: ', data_train.data.shape[1:])



class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
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



############# 对训练数据进行K折划分 #############
def get_k_fold_data(k, i, X, y): 
    '''PARAMS:
    k：折数
    i：第i折作为valid
    X：train_data 训练数据集
    y：train_targets 训练目标
    '''
    '''RETURN:
    返回第i折交叉验证时的训练数据X_train/y_train和验证数据X_valid/y_valid
    '''
    assert k >= 1
    fold_size = X.shape[0] // k  # 每折数据个数 = 总数据条数 / 折数
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start, end)
        # idx为每个valid组的ID
        X_part, y_part = X[idx, :], y[idx]
        if j == i: ### 第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) # dim=0按行叠加
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid

def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

############# K折交叉验证 #############
def k_fold(k, X_train, y_train):
    '''PARAMS:
    k：进行k折交叉验证
    X_train：训练data值
    y_train：训练target值
    '''
    # 定义一些参数
    para_num_epoches = 100
    para_batch_size = 256
    para_optimizer = torch.optim.SGD(Net_model.parameters(), lr = 0.9)
    para_loss_func = nn.CrossEntropyLoss()

    # 记录
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum ,valid_acc_sum = 0, 0
    train_time_sum = 0
        
    for i in range(k):
        X_train1, y_train1, X_valid1, y_valid1 = get_k_fold_data(k, i, X_train, y_train) # 获取k折交叉验证的训练和验证数据
        net =  Net_model ### 实例化Net模型
        
        # 记录时间
        time_start = time.time()
        
        train_ls, valid_ls = train(net=net, train_features=X_train1, train_labels=y_train1, val_features=X_valid1, val_labels=y_valid1, \
                                    num_epochs=para_num_epoches, batch_size=para_batch_size, optimizer=para_optimizer, \
                                    loss_function=para_loss_func, def_init_weight=init_weight)
        
        time_end = time.time()
        
        # 保存train loss和 val loss
        np.savetxt('./loss_figures/第' + str(i+1) + '折训练集的loss', train_ls)
        np.savetxt('./loss_figures/第' + str(i+1) + '折验证集的loss', valid_ls)
        
 
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
        plt.title('第' + str(i+1) + '折训练集与验证集的loss趋势图')
        plt.savefig('./loss_figures/第' + str(i+1) + '折训练集的loss趋势图.png')
        # plt.show()
  
        # 保存模型
        if not os.path.exists('./model_save'):
            os.makedirs('./model_save')	
        torch.save(net, './model_save/Model_K折交叉验证_' +  str(i) + '#MNIST.pt')       
        
        # 记录train消耗时间
        time_train = time_end - time_start
        
        # 某一折的结果打印
        print('*'*25,'第',i+1,'折','*'*25)
        print(' train_time:%.2f seconds\n'%time_train,\
            'train_loss:%.6f'%train_ls[-1][0],'train_acc:%.4f\n'%valid_ls[-1][1],\
              'valid loss:%.6f'%valid_ls[-1][0],'valid_acc:%.4f'%valid_ls[-1][1])
        train_loss_sum += train_ls[-1][0]
        valid_loss_sum += valid_ls[-1][0]
        train_acc_sum += train_ls[-1][1]
        valid_acc_sum += valid_ls[-1][1]
        train_time_sum +=  time_train
    
    
    # 打印最终结果
    print('#'*10,'最终k折交叉验证结果','#'*10) 
    print(' train_time_sum:%.2f seconds\n'%(train_time_sum/k),\
        'train_loss_sum:%.4f'%(train_loss_sum/k),'train_acc_sum:%.4f\n'%(train_acc_sum/k),\
          'valid_loss_sum:%.4f'%(valid_loss_sum/k),'valid_acc_sum:%.4f'%(valid_acc_sum/k))

def celoss_acc(flag,net,x,y, device, loss_func):
    if flag == 1: ### valid 数据集
        net.eval()
        
    x,y, net = x.to(device),y.to(device), net.to(device)
    output = net(x.to(torch.float32))
    result = torch.max(output,1)[1].view(y.size())
    corrects = (result.data == y.data).sum().item()
    accuracy = corrects*100.0/len(y)  #### 5 是 batch_size
    loss = loss_func(output,y)
    net.train()
    
    return (loss.data.item(),accuracy)

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
            X, y = X.to(device), y.to(device)
            output  = net(X.to(torch.float32))
            loss = loss_function(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        # 得到每个epoch的 loss 和 accuracy 
        train_ls.append(celoss_acc(0,net, train_features, train_labels, device=device, loss_func = loss_function))
        if val_labels is not None:
            val_ls.append(celoss_acc(1,net, val_features, val_labels, device=device, loss_func = loss_function))
    
    return train_ls, val_ls

if __name__ == "__main__":
    ### parameters
    k = 1
    X_train = torch.unsqueeze(data_train.data, 1) # shape = [60000, 1, 28, 28]
    y_train = data_train.targets # shape = [60000, 1]

    if __name__ == '__main__':
        k_fold(k=1, X_train=X_train, y_train=y_train)

