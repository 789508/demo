# README
&nbsp; &nbsp;&nbsp; &nbsp;**基于pytorch的多类别图像分类任务实战练习。**
<br/> &nbsp; &nbsp;&nbsp; &nbsp;方便起见，使用torchvision封装好的cifar100数据集进行训练和测试 </br>


## 要求

**环境配置**

- python3.6
- pytorch1.6.0+cu101
- tensorboard 2.2.2(optional)
- numpy1.18.1
- jupyter1.0.0(optional)
- jupyter-client5.3.4(optional)
- jupyter-console6.1.0(optional)
- jupyter-core4.6.1(optional)
- jupyterlab1.2.6(optional)
- jupyterlab-server1.0.6(optional)

## 可用模型

- Alexnet
- Densetion
  - DenseNet121
  - DenseNet169
  - DenseNet201
  - DenseNet264
- GoogleNet
- ResNet
  - ResNet50
  - ResNet101
  - ResNet152
- VggNet
  - Vgg16
  - Vgg19

## 使用指南

### 获取代码
- 1.使用github直接clone到本地环境
- 2.下载压缩包，解压
  - windows解压命令：`unzip demo.zip`
  - Linux解压命令：`sudo unzip FileName.zip `

### 训练&测试网络
- 1.直接使用run.ipynb
- 2.使用终端调用

### arg使用指南
- train
  - net 网络（无默认值） eg.-net vgg16
  - gpu 是否使用GPU（默认不使用）
  - class_num 类别数（默认值100） eg.-class_num 100
  - b batch大小（默认128）
  - warm 是否使用预热（默认为1）
  - lr 学习率（默认为0.1）
  - resume 是否重新开始训练（默认重新训练）
- test
  - net 网络（无默认值） eg.-net vgg16
  - gpu 是否使用GPU（默认不使用）
  - class_num 类别数（默认值100） eg.-class_num 100
  - b batch大小（默认128）
  - weight 测试权重的地址
