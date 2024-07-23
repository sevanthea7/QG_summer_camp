import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.vision.py_transforms as py_trans
import numpy as np
from PIL import Image
import io
import mindspore
from mindspore import Tensor

import mindspore.dataset.vision.py_transforms as py_trans
import numpy as np
from PIL import Image
import io
import mindspore
from mindspore import Tensor


def transform_image(image_bytes):
    # 创建一个 PIL 图像对象
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # 定义图像转换
    transform = [
        py_trans.Resize(size=(224, 224)),  # 调整图像大小
        py_trans.ToTensor(),  # 转换为 Tensor
        py_trans.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 标准化
    ]

    # 应用转换
    for t in transform:
        image = t(image)

    # 由于 `py_trans.ToTensor()` 已经返回了一个 Tensor 对象，你可以直接操作它
    # 添加 batch 维度
    image = image.expand_dims(0)  # 在 MindSpore 中使用 expand_dims

    return image

# 示例用法
with open('../data/test2.png', 'rb') as f:
    image_bytes = f.read()
    processed_image = transform_image(image_bytes)
    print(processed_image.shape)

