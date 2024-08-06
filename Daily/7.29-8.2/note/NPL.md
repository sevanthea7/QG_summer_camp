## 1.Word2Vec

### （1）模型调用参数

#### i. 示例

~~~python
from gensim.models import Word2Vec

word2vec_model = Word2Vec( sentences = x_all, 
                          vector_size = 100, 		# 最终期望提取单词维度的大小
                          window = 5, 				# 窗口大小（周边词+1）
                          min_count = 1, 			# 单词频数小于该值则该单词不参与训练
                          workers = 4
                         )
~~~

其他参数：

`sg: 1(Skip-gram) 0(CBOW) 两个结构`
`hs: 1(hierarchical softmax) 0(negative) 两个优化`

~~~python
def text_to_vector( text ):
    vector = []
    for word in text:
        if word in word2vec_model.wv:					# 检查单词是否在词向量模型的词汇表中
            vector.append( word2vec_model.wv[ word ] )	# 把单词加到词汇表中	
    
    if vector != []:
        return np.mean( vector, axis = 0 )
    else:
        return np.zeros( word2vec_model.vector_size )
~~~

- 检查目的

  eg

  ~~~python
  from gensim.models import Word2Vec
  import numpy as np
  
  sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
  word2vec_model = Word2Vec(sentences, min_count=1)
  
  def text_to_vector(text):
      vector = []
      for word in text:
          if word in word2vec_model.wv:
              vector.append(word2vec_model.wv[word])
      
      if vector != []:
          return np.mean(vector, axis=0)
      else:
          return np.zeros(word2vec_model.vector_size)
  
  
  text = ["cat", "dog", "unicorn"]
  vector = text_to_vector(text)
  ~~~

  - 如果不检查单词是否在词向量模型中，对于 `unicorn` 这样的单词，`word2vec_model.wv['unicorn']` 会导致 KeyError
  - 通过检查，可以忽略 `unicorn`，只计算 `cat` 和 `dog` 的向量的平均值，从而生成一个有效的文本向量

### （2）连续词袋模型

#### i. CBOW

<u>上下文词得到目标词</u>

- 设置上下文窗口长度 -> 通过窗口内的词语预测目标值

- 输入词 -> One-Hot编码 -> embeddings层 -> 词向量

  - 通过onehot编码在矩阵中选定一个特定的行，从embeddings中找到对应的词向量

- 多个上下文统一表示 `v = ( v1 + v2 + v3 + v4 ) / 4` 

  ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407302006605.png)

  

  

- 最终输入softmax函数得到一个预测目标词

#### ii. skip-gram

<u>目标词得到上下文词</u>

- 设置上下文窗口长度 -> 通过窗口内的词语预测目标值
- 使用一个词预测另一个词，要尽量使两个词的词向量在向量空间中尽可能接近，二者点积尽可能大

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407302013973.png)

- 最终返回词表中目标词的概率分布 -> 词汇表中每个词是目标词上下文的可能性

- 词表中的词与目标词的两种关系：

  - 上下文词：标记为1
  - 非上下文词：标记为0

  

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407302021010.png)

## 2.LSTM预测流程

### （1）预处理tsv数据

- 读入tsv文件，分别取为训练集和测试集
- 去除停用词和无意义符号，通过中文分词组件对训练集和测试集分别进行分词
- 处理后的数据作为单个的词分别存入txt文件

### （2）创建数据集

- 读入预处理过数据的txt文件中的每个词转成词向量
- 重新读入tsv文件，不拆分直接打包进dataloader作为训练集和测试集

### （3）训练

- 搭LSTM模型 -> 一层lstm + 一层全连接层
- 存测试准确率最高的模型文件  

### （4）测试

- 把自定义的测试数据进行去停用词操作，转成tensor
- 用训练好的模型进行预测