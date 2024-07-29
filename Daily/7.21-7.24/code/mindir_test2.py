import numpy as np
import mindspore
from mindspore import nn
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
import mindspore.ops as ops
from mindspore import Tensor


class AlexNet(nn.Cell):
    def __init__( self, num_classes, init_weights ):
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(3, 48, kernel_size=11, stride=4, pad_mode='pad', padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 128, kernel_size=5, pad_mode='pad', padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),

            nn.Conv2d(192, 192, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),

            nn.Conv2d(192, 128, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2)
        ])
        self.classifier = nn.SequentialCell([
            nn.Dropout(p=0.5),
            nn.Dense(128 * 7 * 7, 2048),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Dense(2048, 2048),
            nn.ReLU(),

            nn.Dense(2048, num_classes),
        ])
        if init_weights:
            self._initialize_weights()


    def construct(self, x):
        x = self.features(x)
        x = ops.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.HeNormal(mode='fan_out'), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(mindspore.common.initializer.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.Normal(sigma=0.01), cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(mindspore.common.initializer.initializer(
                    'zeros', cell.bias.shape, cell.bias.dtype))


net = AlexNet( num_classes = 5, init_weights = True )

param_dict = load_checkpoint("../model/model3.ckpt")

load_param_into_net( net, param_dict )
input = np.random.uniform(0.0, 1.0, size=[1, 3, 256, 256]).astype(np.float32)
mindspore.export( net, Tensor(input), file_name='flower', file_format='ONNX')

