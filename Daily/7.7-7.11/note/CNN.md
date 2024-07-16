## 1.卷积神经网络基础

#### 1. 卷积神经网络

- CNN
  - 局部连接 -> 图结构是最典型的局部连接结构
  - 权重共享 -> 减少计算量
  - 多层叠加 -> 处理分级模式
- 反向传播算法:BP
- BP算法训练LeNet5模型 -> CNN

#### 2. 全连接层

- 神经元
  - $f（w_1*x_1+w_2*x_2+w_3*x_3+\theta）  → y$
  - $w$：神经元连接权值
  - $\theta$：神经元阈值
  - $f(x)$：激活函数
- 多维特征展平 -> 输出分类回归结果 / 升维降维
- one-hot编码

#### 3. 卷积层

- 滑动窗口 - 局部感知、权值共享
- 目的：进行图像特征提取

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407070914209.png)

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407070917534.png)

- 激活函数

  - 引入非线性，激活/抑制神经元，缓解梯度爆炸

  - digmoid激活函数：梯度值很小，易出现梯度消失

    $f(x)=\frac{1}{1+e^{-x}}$

  - Relu激活函数：反向传播过程中有非常大的梯度经过时可能导**分布中心**小于0，导致该处的倒数始终为0，反向传播无法更新权重失活

    $f(x)=Max(0,x)$

- 更新后的尺寸计算

  $N =\frac{ ( W - F + 2P ) }{S} + 1$

  - W：输入图片的大小W×W
  - F：卷积核的大小F×F
  - S：步长
  - P：延长扩大的像素长度，一般是上下左右对称增加就是$2P$，如果只增加两个方向的，公式中的$2P$应该改成$P$

#### 4. 池化层

- 目的：对特征图进行稀疏处理，减少运算数据量
- 没有训练参数，只改变w和h，不改变深度
- MaxPooling下采样层 - 找最大值

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407070933443.png)

- AveragePooling下采样层 - 求平均值

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407070936234.png)

#### 5.误差计算

##### 1.针对多分类问题 -> softmax输出

$H = - \sum_io_i^*log(o_i)$

softmax处理目的：满足概率计算

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407070944294.png)

##### 2.针对二分类问题 -> sigmoid输出

概率和不为1

$H = - \frac{1}{N} \sum_{i=1}^{N}[o_i^*log(o_i) + (1 - o^*_i)log(1 - o^*_i)]$

**默认log为ln** 倒数为1/n

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407070958897.png)



#### 6.权重更新

- SGD优化器
  - $w_{t+1} = w_{t} + 学习率*g(w_t)$
    - $g(w_t)$为t时刻对参数$w_t$的损失梯度
    - 易受样本噪声影响、可能陷入局部最优解

  - 优化：SGD+Momentum优化器

    $\nu_t = \eta*\nu+学习率*g(w_t)\\w_{t+1} =  w_t - \nu_t$

- Agagrad优化器

  - $s_t = s_{t-1}+g(w_t)*g(w_t)\\w_{t+1} = w_T - \frac{学习率}{s_t + \epsilon }*g(w_t)$
    - $\epsilon$是一个很小的数，防止分母为0
    - 学习率下降太快可能还没收敛就停止训练

  - 优化：RMSProp优化器

    $s_t = \eta * s_{t-1}+(1 - \eta) * g(w_t)*g(w_t)\\w_{t+1} = w_T - \frac{学习率}{s_t + \epsilon }*g(w_t)$

- Adam优化器

  ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407071016994.png)

#### 7.迁移学习

常见方式

1. 载入权重后训练所有参数
2. 载入权重后之训练最后几层参数
3. 载入权重后在原网络上自己加一层全连接层，只训练自己加的这一层参数



## 2.Lenet

**导入本地数据集**

1. 文件换成浏览器格式

   ~~~
   file//+文件地址
   ~~~

2. 在cifar.py文件中将**url**改成本地文件的浏览器地址`"C:\Users\EC319\AppData\Roaming\Python\Python311\site-packages\torchvision\datasets\cifar.py"`

3. ~~~
   trainset = torchvision.datasets.CIFAR10( root = './data', train = True, download = True, transform = transform )
   ~~~

#### 1.创建一个继承于nn.Module的类

- `__init__`

  - Conv2d：（self, 特征矩阵深，卷积核个，卷积核大小）
  - MaxPool2d：最大下采样（卷积核个数，卷积核大小）

  $N =\frac{ ( W - F + 2P ) }{S} + 1$

  eg`32*32`深度为3图片，W = 32，F=5（边长为5的卷积核），P=0（没有延长边），  S=1（步长默认为1），得到输出大小为N=28 → `28*28`深度为16

- `forward`

  结合init的输出结果和激活函数，得到一个展平的一维向量

  view函数（ -1， 总元素数） -> 展平成一维

#### 2.创建训练集测试集

- 导入训练集`torchvision.datasets.xxxx( root= , train= False,  download = True, transform = transform`

  - 数据预处理transforms.Compose

    transforms.ToTensor() -> 把图片从`高度*宽度*深度`[0,255]转化成`深度*高度*宽度`的torch tensor[0.0, 1.0]

  - transform.Normalize

    归一化 -> 加快训练速度
    
    eg. transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)) 三个通道都归一化，分别是mean和std，输出为（输入- mean）/std，即（输入-0.5）/0.5

#### 3.训练

1）实例化模型，选择损失函数和优化器

2）迭代训练集：两个for循环

​		for -> 完整训练的次数

​				for 遍历数据集

- loss 计算输出和标签之间的损失

- loss.backwards 反向传播，计算梯度

- running_loss += loss.item

  - loss = torch.tensor(2.5)
    loss_value = loss.item()  # loss_value = 2.5

    张量转化成python数值

- accuracy：

  eg.`predict_y = torch.tensor([1, 0, 1, 1])`

  `test_labels = torch.tensor([1, 1, 1, 0])`

  执行

  1. `torch.eq(predict_y, test_labels)` -> `tensor([True, False, True, False])`
  2. `.sum()` -> `tensor(2)` （预测正确的数量为 2）
  3. `.item()` -> `2`（转换为 Python 数值）
  4. `test_labels.size(0)` -> `4`（测试样本的数量）
  5. 计算准确率 `accuracy = 2 / 4 = 0.5`

#### 4.预测

​		1）将输入的图像转化成`32*32`，并进行预处理

​		2）实例化对象.load_state_dict( torch.load( 图片地址 ）

​		3）Image.open(图片） -> 添加新的一列batch

​		4）输出概率最大的分类类别

## 3.AlexNet

每一层随机忽略一些节点来减少过拟合现象

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407082201682.png)
$$
N =\frac{ ( W - F + 2P ) }{S} + 1
$$

#### 1.创建一个继承于nn.Module的类

- 卷积层
- 随机失活
- 初始化权重

#### 2.导入数据集 -> 在程序外已划分好训练集数据集

- 随机裁剪成大小一样（randomresizecrop / resize）-> totensor -> normalize
- ImageFolder
- list -> dataset.class_to_idx

#### 3.训练

- 实例化、选择损失函数、选择优化器

- for

  xxx.train()

  running_loss = 0.0

  for

  ​	xx, xx = data

  ​	optimizer.zero_grad()

  ​	loss = loss_f( xx, xx )

  ​	loss.backward()

  ​	optimizer_step()

  ​	running_loss = loss.item()

  xxx.eval()

  with torch.no_grad():

  ​	for...

#### 4.预测

- 修改大小（resize）-> totensor -> normalize

- Image.open()
- net.load_state_dict( torch.load() )
- 预测

## 4.VGG

#### 1.CNN感受野 -> 减少训练使用的参数

计算公式：
$$
F(i) = ( F(i+1) - 1 ) * S + K
$$
$F(i)$：第 $i$ 层的感受野

$S$：第 $i$ 层步距

$K$：第 $i$ 层卷积核的大小

e.g.:

（4）- > F = 1

Conv3×3(3): F = ( 1 - 1 ) × 1 + 3 = 3 

Conv3×3(2):  F = ( 3 - 1 ) × 1 + 3 = 5

Conv3×3(1): F = ( 5 - 1 ) × 1 + 3 = 7

三层3×3的卷积核最终得到的卷积核的感受野相当于7×7的大小 -> 目的：减少训练需要参数的个数($49C^2 : 27C^2$ )

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407091548186.png)



#### 2.一次创建不同层数的网络结构

![1720514020258](C:/Users/EC319/AppData/Roaming/Typora/typora-user-images/1720514020258.png)

↓    通过字典转化成四个不同的网络

~~~python
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
~~~





## 5.GoogLeNet

#### 1.卷积降维

e.g.：深度为512，用64个5×5的卷积核进行卷积 -> $5*5*512*64=819200$

降维：深度为512，先用24个1×1的卷积核进行卷积 -> 深度变为24，再用64个5×5的卷积核进行卷积 -> $1*1*512*24 + 5*5*24*64=12288+38400=50688$

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407091617447.png)

~~~python
torch.cat( [ x, y ], dim = 1 )  # -> 深度方向上拼接
~~~



#### 2.辅助分类器

训练使用，验证不使用 ->   通过参数来屏蔽

#### 3.损失计算

~~~python
logits, aux_logits2, aux_logits1 = net( images.to( device ) )
# 真实输出， 辅助分类器1输出，辅助分类器2输出
loss0 = loss_f( logits, labels.to( device ) )
loss1 = loss_f( aux_logits1, labels.to( device ) )
loss2 = loss_f( aux_logits2, labels.to( device ) )
loss = loss0 + loss1 * 0.3 + loss2 * 0.3	# 按一定比例相加
~~~





## 6.ResNet

- 超深的网络结构，解决退化问题，层数越多效果越好
- residual
- batch normalization -> 不用drop out

#### 1.residual

- 相加（GoogLeNet是拼接）
- ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407100843088.png)

*层数少的结构2层，多的结构有3层 -> 实际应用上需要**用参数调整第三层的深度**

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407092007247.png)

用1×1的卷积核来降维和升维

#### 2.Batch Normalization

- feature map -> 均值为0，方差为1

- 训练时使用，用布尔参数控制`model.train()``model.eval()` 来控制
  - `.train()`：启用batch normalization（保证BN层能用到每一批数据的均值和方差） 和 dropout（随机选取的参数）
  - `.eval()`：不启用batch normalization和dropout -> 测试时使用
- BN层放在卷积层（Conv）和激活层（eg. ReLU）之间
- 不需要使用bias

## 7.ResNeXt

- 组卷积：分成n部分，对每个组卷积

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407092052201.png)

对应：上面一节对应第一个feature map，下面一节对应第二个feature map，每个位置两个feature map的元素与kernel的乘积之和为合成的feature map该位置的元素

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407100914473.png)

最终结果在数学上完全等价

block在层数>=3时效果较好





## 8.MobileNet

- 大大减小运算量、参数量 

### （1）V1

轻量级CNN网络 -> 移动端&嵌入式设备

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407101056753.png)

DW卷积

| 传统卷积                      | DW卷积                                           |
| ----------------------------- | ------------------------------------------------ |
| 卷积核深度 = 输入特征矩阵深度 | 卷积核深度 = 1                                   |
| 输出特征矩阵深度 = 卷积核个数 | 输入特征矩阵深度 = 卷积核个数 = 输出特征矩阵深度 |

DW：每个卷积核负责一个输入矩阵的深度

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407092129933.png)

M：输入矩阵的深度

N：输出矩阵的深度

$D_F$：输入矩阵的长宽

$D_K$：卷积核的长宽

DK：$D_K * D_K * M * D_F * D_F$

DF：$M*N*D_F*D_F$

### （2）V2

ReLU6 激活函数

ReLU激活函数会对低维特征信息造成较大的损失 -> 找线性激活函数来减小这些损失

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407092142603.png)

==shortcut连接存在的前提条件：stride = 1 且输入特征矩阵与输出特征矩阵shape相同时==

### （3）V3

加入了注意力机制 & 激活函数的更新

1.注意力机制 -> 权重

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407092159791.png)

求出两个矩阵的均值 -> 经过两个全连接层 -> 新权值 -> 对应乘到两个矩阵上

2.重新设计耗时层结构

- 减少第一个卷积层的卷积核个数 32 -> 16
- 精简last stage

3.重新设计激活函数

## 9.ShuffleNet

### （1）V1 

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407101116656.png)

## 10.EfficientNet

- 探索输入分辨率，网络的深度、宽度的影响
  - 增加网络深度
    - 可以得到更加丰富、复杂的特征并很好的应用到其他任务中
    - 深度过深会面临梯度消失、训练困难的问题
  - 增加网络宽度
    - 可以得到更加细粒度、更容易训练
    - 很难学习更深层次的特征
  - 增加输入分辨率
    - 可以得到更高细粒度的特征模板
    - 准确率效益也会减少，增加了计算量
- ->  同时增加输入分辨率，网络的深度、宽度

