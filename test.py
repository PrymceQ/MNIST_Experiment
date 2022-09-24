import torch
from torch import nn
import numpy as np
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

X_test = torch.unsqueeze(data_test.data, 1) # shape = [60000, 1, 28, 28]
y_test = data_test.targets # shape = [60000, 1]


# 读取保存的模型
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
net_test = torch.load('.\model_save\Model_#MNIST.pt').to(device=torch.device('cpu'))

test_pred = net_test(X_test.to(torch.float32))
result = torch.max(test_pred, 1)[1].view(y_test.size())
corrects = (result.data == y_test.data).sum().item()
accuracy = corrects*100.0/len(X_test)
loss_func = nn.CrossEntropyLoss()
loss = loss_func(test_pred, y_test).data.numpy()

print(' Cross Entropy Loss: ', round(float(loss), 4), '\n',
      'Accuracy: ', accuracy)

# 将测试集中错误预测的图片进行展示
wrong_predict_idx = (result.data != y_test.data).nonzero(as_tuple=False).flatten()[:24]
plt.figure()
k = 0
for j in wrong_predict_idx:
    k+=1
    plt.subplot(4,6,k),plt.imshow(data_test.data[j].numpy(), cmap='gray')
    
    plt.title('target: %i' % y_test.data[j].numpy() + '\n' +  
              'predict: %i' % result.data[j].numpy())
    plt.xticks([]),plt.yticks([])
    plt.tight_layout()
    plt.savefig('./visualization_figures_save/wrong_predicted.png')
plt.show()

for parameter in net_test.parameters():
    print(parameter)
