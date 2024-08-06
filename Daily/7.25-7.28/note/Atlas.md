https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC1/Application%20Development%20Guide/msadgp/msadgp_0005.html

密码：Mind@123

配置环境变量：

~~~python
. /usr/local/Ascend/mxVision/set_env.sh
~~~

~~~
atc --model=model.onnx --framework=5 --output=model2 --soc_version=Ascend310B4 
~~~



~~~python
resnet50_sdk_python_sample
├── data
│ ├── test.jpg               # 测试图片
├── model
│ ├── resnet50.om            # ResNet-50网络的om模型
├── utils
│ ├── resnet50.cfg           # 用于后处理的配置文件，包含类别数量和是否包含softmax操作
│ ├── resnet50_clsidx_to_labels.names          # 类别标签文件
├── main.py                   # 运行程序的脚本
~~~

~~~
atc --model=model.onnx --framework=5 --output=model --soc_version=Ascend310B4
~~~

0:Caffe; 1:MindSpore; 3:Tensorflow; 5:ONNX

> ATC转换过程中别开vpn



**图像识别评价指标**

- precision 精确率 P=预测为正样本的结果/预测为正样本的所有结果
- recall 召回率 R=预测为正样本的结果/实际为正样本的所有样本
- ACC 召回率 R=预测为正样本的结果/实际为正样本的所有样本

- mAP
  - mAP (Mean Average Precision)：
    mAP 是所有类别的平均精度均值。它是通过计算每个类别的 Average Precision (AP)，然后对这些 AP 值取平均得到的。AP 是一个类别的 Precision-Recall 曲线下的面积。在目标检测任务中，mAP 衡量的是模型对于不同对象类别的检测精度和召回率的综合性能。
  - mAP_50：
    mAP_50 是在 Intersection over Union (IoU) 阈值为 0.5 时的 mAP。IoU 是一个衡量预测边界框和真实边界框重叠程度的指标。mAP_50 特别关注当预测边界框与真实边界框有较高重叠时模型的性能。
  - mAP_75：
    mAP_75 是在 IoU 阈值为 0.75 时的 mAP。这个指标更加严格，因为它只考虑那些预测边界框与真实边界框有很高重叠的情况。mAP_75 通常比 mAP_50 低，因为它的评估标准更加严格。

- IOU 交并比

  ![](https://sevanthea7.oss-cn-beijing.aliyuncs.com/QGworks/202407260948661.png)