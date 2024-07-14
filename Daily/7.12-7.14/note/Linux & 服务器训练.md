## Linux

-> 车队学习任务文档内容

**1、** **ubuntu界面截图：**

 ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407141504050.png)

 

**2、** **写出下列操作的命令：**

**a)** **移动a文件夹至/c/b**

mv ./a ./c/b

**b)** **解压缩a.tar文件：**

tar -xvf a.tar

**c)** **显示当前目录所有文件：**

ls -a

**d)** **使用apt工具安装Google** **chrome：**

sudo apt-get install Google chrome

**e)** **文本编辑器打开.bashrc：**

vim ~/.bashrc

**3、** **写出下列命令的含义：**

**a)** **touch a.txt：**

如果不存在a.txt文件，新建一个a.txt空文件。

如果存在a.txt文件，改变文件的末次修改日期。

**b)** **grep -rn "AB"：** 

查找文件夹中包含“AB“字符的文件，并显示它所在的行号。

**c)** **ifconfig：** 

网络配置工具，用于配置和显示网络接口的具体状况、启用和禁用网络接口、调试网络问题，需要sudo权限。

**d)** **echo** **a** **>> /home/spring/b.txt：** 

在spring目录下的b.txt文件内容后面追加“a“。

**e)** **cat aa.txt bb.txt：**

按顺序显示aa.txt和bb.txt的文件内容。







## 服务器训练

https://blog.csdn.net/LWD19981223/article/details/127085811?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172094161816800207036345%2522%252C%2522scm%2522%253A%252220140713.130102334

#### 1.AutoDL 创建实例

- 选择需要的配置环境
- 上传数据 -> AutoDL自带网盘 / 直接上传（ zip压缩包 ）/ Xftp软件

#### 2.配置环境

~~~
vim ~/.bashrc
i														# 编辑
source /root/miniconda3/etc/profile.d/conda.sh			# 最后一行加上环境路径
esc -> wq												# 保存并退出
bash													# 刷新
conda activate base
conda create -n py39 python=3.9							# 创建新环境
conda activate py39
安装torch、torchvision... # -> 要先下载镜像 后再autodl-nas/ 目录下
~~~

-> 进入py38环境

~~~
conda install ipykernel
ipython kernel install --user --name=py39
~~~

#### 3.数据上传

**Xftp**

https://blog.csdn.net/qq_45073592/article/details/133895731?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172094161816800207036345%2522%252C%2522scm%2522%253A%252220140713.130102334

更改环境

python -> project interpreter -> on SSH