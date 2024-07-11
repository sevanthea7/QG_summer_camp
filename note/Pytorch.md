### （一）导入数据集

~~~python
from torch.utils.data import Dataset, Dataloader
~~~

~~~python
class MyDataset( Dataset ):
    def __init__( sekf, file ):
        self.data = ...
    def -> 返回data
    def -> 返回dataset大小
~~~

**dataset**  = MyDataset( file )

dataloader = Dataloader( **dataset**, batch_siza = 5, shuffle = False )

-> 把五个batch拼成一个

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407081009157.png)

### （二）基本语法

##### 1.创建数组

（1）直接输入数字/numpy数组

~~~python
x = torch.tensor( [ [1, -1],
				  [-1, 1] ] )
x = torch.from_numpy( np.array( [ [1, -1],
				  [-1, 1] ] ) )
~~~

（2）创建全是0/1的数组

~~~python
x = torch.zeros( [2, 2] )  # -> 2*2的零矩阵
下= toch.ones( [1, 2 ,5] )
~~~

2.矩阵转换

Transform

~~~python
x = torch.zeros( [2, 3] )
x = x.transpose( 0, 1 )  # -> size: 3, 2
~~~

其他：reshape、view...

##### 3.消去/添加维度

**Squeeze**

sqeeze()：消除维度为1（没用的）维度

~~~python 
x = torch.zeros( [1, 2, 3] )
x = x.squeeze( 0 )	# -> 消去dim = 0 size = 2, 3
~~~

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407081018735.png)

**Unsqueeze**

~~~python
x = torch.zeros( [ 2, 3 ] )
x = x.unsqueeze( 1 ) # -> dim = 1 size: 2, 1, 3
~~~

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407081020242.png)

##### 4.拼接tensor

~~~ python
x = torch.zeros( [2, 1, 3] )
y = torch.zeros( [2, 3, 3] )
z = torch.cat( [x, y, z], dim = 1 ) # -> 沿第一维度拼接
 # -> size: 2, 6, 3
~~~

![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407081023249.png)



~~~python
x = torch.tensor( [[],[]], requires_grad=True ) # 计算梯度
z = x.pow(2).sum()
z.backward()

~~~

$z = \sum\sum x^2_{ij}$



### 三. torch.nn

`import torch.nn as nn`

##### 1.线性

Linear Layer -> $w*x + b$

~~~python
layer = torch.nn.Linear( 32, 64 )  # -> 输入32个，输出64个
layer.weight.shape
layer.bias.shape
~~~

##### 2.非线性

~~~
nn.Sigmoid
nn.Relu
~~~

eg

~~~python
class MyModel( nn.Module ):
	def __init__( self ):
        super（ MyModule, self ).__init__()
        self.net = nn.Sequential(
            nn.Linear( 10, 32 )
            nn.Sigmoid()
            nn.Linear( 32, 1 )
        )
        '''
        self.layer1 = nn.Linear( 10, 32 )
        self.layer2 = nn.Sigmoid()
        self.layer3 = nn.Linear( 32, 1 )
        '''
    def forward( self, x ):
    	return self.net( x )
    	'''
    	out = self.layer1( x )
    	out = self.layer2( out )
    	return self.layer3( out )
    	'''
~~~

##### 3.误差函数

~~~python
loss_f = nn.MSELoss()	# -> MSE
loss_f = nn.CrossEntropyLoss()

#调用
loss = loss_f( model_out, expect )
~~~

### 四.torch.optim

eg

~~~python
torch.optim.SGD( model.parameters(), lr = 0.01, momentum =  0 )
~~~

- 每轮batch必要流程

  ~~~python
  optimizer.zero_grad()	# 清空历史梯度
  ...
  loss.backward()			# 反向传播
  ...
  optimizer.step()		# 调整参数
  ~~~

### 五. 存储调用模型

##### 1.存储

~~~python
torch.save( model.state.dict(), path )
~~~

##### 2.调用

~~~python
pth = toch.load( path )
model.load_state_dict( pth )
~~~







