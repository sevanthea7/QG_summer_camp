## GAN

<u>无监督</u>

#### 1.生成器

- 生成新的样本数据
- 输入随机噪声输出高维向量（图片文本语音）

#### 2.判别器

- 区分输入的数据是真实样本还是生成器生成的
- 输入高维向量，输出表示数据真实性概率的标量

#### 3.对抗生成

- 生成器、判别器交替训练，提高判别器的区分能力、提高生成器的生成能力
- 生成器生成的数据逐渐接近真实数据、判别器无法准确分别时。达到收敛

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407310922759.jpeg)

#### 4.应用

- 文本到图像
  - 文本编码
  - 生成器：根据编码输出相应的图像，判别器：判定图像是否与文本描述相符
  - 对抗训练：区分生成图像和真实图像

- 图像到图像
  - 生成器：将图源转换为目标风格的图像，判别器：区分生成的图像和真实目标风格的图像
  - 损失
    - 循环一致性：训练两个生成器，分别为从图源转到目标域、从目标域转回源域 -> 确保经过两次转换后的图像与原始图像的差异最小化，保证在没有陈规训练数据的情况下学习有效的映射
    - 对抗性：区分生成图像和真实图像

#### 5.改进

- CGAN 
  - 把无监督的GAN变成半监督/有监督的模型
  - 在生成器和判别器中引入变量条件y（可以是label或其他数据形式），将y和GAN原有的输入合并为一个向量作为CGAN的输入

- DCGAN
  - 加入反卷积 -> 将输入的噪声变得越来越大
  - 用于文本转图像 -> 除了判别样本的真实性还要判断与文义是否相符

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407311038932.png)

## VAE

### 1.Auto-Encoder

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202408010925105.png)

- Encoder：将开始的数据压缩为低维向量
- Decoder：把低维向量还原为原来的数据

### 2.Variational AutoEncoder

- 用自编码器去产生很多高斯分布，去拟合样本的分布，然后某个x对应的高斯分布里采样z，然后复原成x
- 图片的特征向量z采样于某种高斯分布，我们希望这个分布贴近标准正太分布，然后通过编码器生成对应均值和方差采样z，希望z能复原图片所以去找这个z背后的高斯分布，这个高斯分布的均值就是最大概率生成特征z，可以复原图片（在数据集如果有2个图片的特征分布都在这个点有所重合的话，可能产生的是2个图片中间的插值图片）

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202408010929568.png)

#### （1）与AE的对比

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202408010930489.png)

- AE
  - 直接的一对一关系，在找不到对应关系时decoder会输出随机的内容
  - 只能重构输入的数据，产生尽可能接近原来的数据或者原来的数据本身
- VAE
  - 在code中添加噪声，可以让在满月对应的噪声范围内都可以转换成满月，弦月对应的噪声范围内也可以转换成弦月
  - 在不是噪声的部分输出的内容可能时介于满月和弦月之间的图 -> 即可以输出数据中不包含的数据

隐藏层：

- VAE隐藏层服从高斯分布
- AE中隐藏层分布无分布要求

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202408010952612.png)

$m_i$：原来的code

$c_i$：带噪声的code

$\sigma_i$：决定噪声大小 -> 自己学习生成的，需要用公式限制不能太小

vector z -> 每个维度代表一个特征

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202408011011369.png)

- 问题 -> 模仿，没有产生新的图片

#### （2）高斯混合模型

- K个子分布组成的混合分布 -> 表示了观测数据在总体中的概率分布

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202408011038863.png)

红色曲线为整个数据的高斯混合分布，蓝色曲线为组成红色曲线的单个高斯模型分布
$$
P(x) = \sum^K_1P(z)P(x|z)
$$
$K$：一共由k个子模型组成

$P(z)$：表示观测数据属于第z个子模型的概率，$P(z) \geq 0 且 \sum^K_1P(z)P(x|z)$

$P(x|z)$：z个子模型在样本x的分布

#### （3）EM算法

E步和M步的交替进行，直至收敛