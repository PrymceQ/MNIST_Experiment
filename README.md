# MNIST_Experiment
Introduction Experiment of MNIST Dataset

>MNIST数据集下载链接：http://yann.lecun.com/exdb/mnist/

将下载后的数据集移至`./data/MNIST/raw/`

## data_visualization.ipynb

1. 用来进行数据的各种可视化，并且将输出的图片保存于路径 `./visualization_figures_save` 下。

## train.py

不带交叉验证的训练文件。直接修改
    L102    网络结构（其中为LeNet配置）
    L123    超参数设置

如果遇到GPU内存问题，那是因为在训练过程中在做评价的时候，我们把所有的数据（60000+10000条）都导入了GPU造成的。（这个问题一般不会出现，除非扩大了网络结构）问题出现后涉及代码L75-L80

## train_k折交叉验证.py

1. 用来进行模型的训练及训练过程的保存

   L110-111    模型.pt文件保存路径 `./model_save`

2. train_k折交叉验证.py文件可以进行修改的地方。
   L35-48      Net_model   1*28*28 -> 1*10
   L93         num_epoches 训练轮数
   L94         optimizer   参数优化方法
   L95         loss_func   计算损失方法，分类用cross entropy loss
   L193        k           k折交叉验证

## test.py

1. 用来使用训练完的模型pt文件对测试集数据进行推理，并且将测试集中的预测错误的例子前16个进行展示。

   pt文件保存位置 `./model_save`。


2. test.py文件可以进行修改的地方。

   L31     pt文件的路径可以按照需要修改。
