## Attention

$$
Q = W_qX\\K = W_kX\\V=W_vX\\A = K^T\\A'=softmax(A)\\Y=VA'
$$

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407121937600.png)

$\alpha ’$ 相当于向量 $v$ 的权重，$\alpha$ 越大代表对应向量的影响越大、越重要

- $q$ 词的“查询”向量
- $k$ 词的“被查”向量
- $v$ 词的“内容”向量

## Transformer

- encoder & decoder
  - 都是六层 
  - 每一个decoder都会接受最后一个encoder的输出。
  - ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407122131057.png)

- attention
  - 向量间相互交流、不断更新
- 多层感知器
  - 向量不相互交流
- 不断重复这两种层

### 1.Encoder

#### （1）位置编码

确定不同字符的位置顺序 -> 奇数位置/偶数位置

偶数位置：$PE_{pos, i} = sin( pos/10000^{2i/d_{model}})$

奇数位置：$PE(pos, i) = cos( pos/10000^{2i/d_{model}})$

$pos$：表示一句话中的第几个字

$i$：字向量的第 $i$ 维度

$d_{model}$：字向量一共有多少维

对每个位置的顺序进行编码

eg. 第一个 -> $PE_{1,2} = sin( 1/10000^{2*2/4})$

第二个 -> $PE_{2,3} = cos( 2/10000^{2*3/4})$

#### （2）self -attention

-> 更容易捕获句子中长距离的相互依赖

1. embedding，输入单词转为词向量
2. 根据嵌入向量利用矩阵乘法得到q、k、v三个向量
3. 为每一个向量计算相关性：$q ⋅ k^T$
4. 为了梯度的稳定，除以$\sqrt{d_k}$
5. 进行softmax归一化得到权重系数
6. 与 $v$ 点乘得到加权的每个输入向量的评分
7. 相加之后得到最终的输出结果$z = \sum$

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407121950189.png)

eg.: $\alpha_{1,2} = q^1·k^2$

$Q · K^T$ 得到一个 $a_{i，j}$ 组成的矩阵 -> $A$ 矩阵乘上 $V$ 矩阵 $( v_1, v_2, v_3, v_4 )$  得到 $b _n$ 

##### Multi-head Self-attention

$q^{i,1}, q^{i,2}$

##### Truncated Self-attention

只参考一定范围内的信息

#### （3）Add & Norm

每经过一个层都要做一次add&norm的操作

残差：add （residual）-> 经过self-attention得到的向量和原向量相加得到新的向量

​			norm -> noramalization标准化

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407121953306.png)

#### （4）FeedForward

相当于全连接层，完成之后也要add&norm

![1720786612054](C:/Users/EC319/AppData/Roaming/Typora/typora-user-images/1720786612054.png)

### 2.Decoder

#### 1.Masked-Attention

把未知的数的 $\alpha$ 设置成负无穷，让它们的影响最小

 ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407131108377.png)

#### 2.Cross-Attention

eg. 用 $q^1$ 进行查询

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407122015170.png)



#### 3.Stop Token