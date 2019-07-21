# CNNClassifier

### 1.总体介绍

最近在之前细粒度分类任务所用的代码Trainer的基础上，完善了CNNClassifier这一个专门用于图像分类任务的工具类，用于快速进行图像分类任务和图像分类任务的迁移学习。

CNNClassifier只需要一行代码就能够调用分类模型，定义评估指标、优化器、损失函数以及学习率调度。

例如：

```python
trainer = CNNClassifier(model='resnet18', data_loader=train_loader, lr=[0.01,0.1],classes=10,val_loader=val_loader, pretrained=True,
lr_scheduler='MultiStepLR', lr_step=50, weight_decay=5e-4,
metrics = {'accuracy': accuracy},lr_param={'milestones':[10, 15]},
optimizer='SGD', optim_param={'momentum':0.9})
```

上述代码定义了一个resnet18的模型，且用于10类别分类任务，学习率范围是[0.01,0.1]，学习率调度是

MultiStepLR，优化器是SGD，动量是0.9，weight_decay=5e-4。

针对迁移学习，CNNClassifier也有freeze函数和unfreeze函数。通过freeze函数冻结特征层的权重，只训练分类层的权重，可以加快模型训练。unfreeze函数用于解冻权重，来进行全局微调。

迁移学习的例子：

```python
trainer.freeze(lr=0.01) #冻结权重,设置一个较大的lr
trainer.train(20) #训练20epoch
trainer.unfreeze(lr=0.001) # 设置一个较小的lr
trainer.train(20) #再训练20个epoch
```

同时该CNNClassifier已经内置了torchvision的pretrained models，同时可以通过定义classes来确定分类任务类别个数。其中的resnet模型还支持非224x224大小的图像数据集

### 2.项目的总体架构：

```
Pytorch_Trainer/
|-- Data/
|-- Data_loader/
|-- Ensemble/
|-- LR_scheduler/
|-- Pretrained_Models/
|-- Trainer/
|-- Utils/
|-- main.py
|-- README
```

Data: 用于存储训练数据集

Data_loader: 用于放置图像数据集类已经图像transforms

Ensemble： 相关的用于集成学习的工具类

LR_scheduler： 学习率调度的工具类，包括SGDR和CyclicLR这两个带重启的学习率调度工具，以及一个用于估计最佳学习率的函数。

Pretrained_Models： 用于调用预训练的模型的工具类，包括了大部分的torchvison.models。

Trainer：CNNClassifier的定义处

Utils：包括一些工具方法，例如图像查看，logger，metrics的定义以及一些数据集的预处理

mian.py 整个项目的入口

### 3.CNNClassifer的输入参数详解：

3.1 支持的预训练的模型有：

'vgg16','vgg19','AlexNet','resnet18','resnet34','resnet50', 'resnet101', 'densenet121', 'densenet161', 'se_resnext101_32x4d', 'se_resnet101'

实际中支持更多的模型，可以修改通过Pretrained_Models底下的MyPretrainModels来达到添加更多模型的作用。**该功能主要torchvision.models 和 pretrained-models.pytorch支持**。

可以通过model参数来定义模型类型，通过pretrained系数来定义是否需要预先训练模型。

通过 classes参数来定义分类任务的类别个数，来调整预训练模型的分类层的网络结构

3.2 优化器

由于我个人常用SGD+momentum和Adam，暂时只支持SGD和Adam。

可以通过optimizer来定义优化器类型，通过lr来传递学习率参数，通过optim_param来传递优化器额外的参数，如momentum。通过weight_decay来定义权重衰减系数。

3.3 lr调度

通过lr_scheduler来定义lr学习率调用，支持的有'Step_LR','SGDR','CyclicLR', 'MultiStepLR'。通过 lr_step来定义lr的step,,lr_param用于补充额外的lr调度的参数，如MultiStepLR的'milestones'。

由于对于SGR和CyclicLR需要一个lr的区间，以此lr默认是一个2元素的数组

3.4 数据调用

data_loader是训练集的数据加载器

val_loader则是验证集的数据加载 器

### 4.CNNClassifier的主要类方法：

def freeze(self, lr=None):冻结特征层的权重，可以通过lr参数来调整lr

def unfreeze(self, lr=None):解冻特征层的权重，可以通过lr参数来调整lr

def train(self, Epochs=1):训练模型，Epochs用于定义训练的epoch数量

def evaluate(self, data_loader):用于调用模型，不训练。data_loader是数据集的loader

def save(self, output_path, file_name):用于保存模型，output_path是保存文件夹，file_name是文件名

def load(self, model_path):用于加载保存的模型。model_path是文件的地址
def model_init(self):用于初始化模型参数

## 5.refernce

fastai项目

cs231的pytorch 项目

pretrained-models.pytorch项目