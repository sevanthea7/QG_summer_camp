import numpy as np  # 用于对多维数组进行计算
import cv2  # 图片处理三方库，用于对图片进行前后处理

from mindx.sdk import Tensor  # mxVision 中的 Tensor 数据结构
from mindx.sdk import base  # mxVision 推理接口
from mindx.sdk.base import post  # post.Resnet50PostProcess 为 resnet50 后处理接口


'''初始化资源和变量'''
base.mx_init()  # 初始化 mxVision 资源
pic_path = 'data/test.jpg'  # 单张图片
model_path = "model/resnet50.om"  # 模型路径
device_id = 0  # 指定运算的Device
config_path='utils/resnet50.cfg'  # 后处理配置文件
label_path='utils/resnet50_clsidx_to_labels.names'  # 类别标签文件
img_size = 256

'''前处理'''
img_bgr = cv2.imread(pic_path)
img_rgb = img_bgr[:,:,::-1]
img = cv2.resize(img_rgb, (img_size, img_size))  # 缩放到目标大小
hw_off = (img_size - 224) // 2  # 对图片进行切分，取中间区域
crop_img = img[hw_off:img_size - hw_off, hw_off:img_size - hw_off, :]
img = crop_img.astype("float32")  # 转为 float32 数据类型
img[:, :, 0] -= 104  # 常数 104,117,123 用于将图像转换到Caffe模型需要的颜色空间
img[:, :, 1] -= 117
img[:, :, 2] -= 123
img = np.expand_dims(img, axis=0)  # 扩展第一维度，适应模型输入
img = img.transpose([0, 3, 1, 2])  # 将 (batch,height,width,channels) 转为 (batch,channels,height,width)
img = np.ascontiguousarray(img)  # 将内存连续排列
img = Tensor(img) # 将numpy转为转为Tensor类

'''模型推理'''
model = base.model(modelPath=model_path, deviceId=device_id)  # 初始化 base.model 类
output = model.infer([img])[0]  # 执行推理。输入数据类型：List[base.Tensor]， 返回模型推理输出的 List[base.Tensor]

'''后处理'''
postprocessor = post.Resnet50PostProcess(config_path=config_path, label_path=label_path)  # 获取后处理对象
pred = postprocessor.process([output])[0][0]  # 利用sdk接口进行后处理，pred：<ClassInfo classId=... confidence=... className=...>
confidence = pred.confidence  # 获取类别置信度
className = pred.className  # 获取类别名称
print('{}: {}'.format(className, confidence))  # 打印出结果  

'''保存推理图片'''
img_res = cv2.putText(img_bgr, f'{className}: {confidence:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)  # 将预测的类别与置信度添加到图片
cv2.imwrite('result.png', img_res)
print('save infer result success')
