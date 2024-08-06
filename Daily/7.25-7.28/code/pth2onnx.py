import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(Net, self).__init__()
        self.method = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # in: 3, 224, 224   (224-11+2)/4 + 1 = (为了保证是整数所以选择性删去最后一行) out: 48, 55, 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55-3)/2 + 1 = 27     out: 48, 27, 27

            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # out: 128, 27, 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27-5+2)/2 + 1 = 13   out: 128, 13, 13

            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # out: 192, 13, 13
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # out: 192, 13, 13
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # out: 128, 13, 13
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)  # (13-3)/2 +1 = 6       out: 128, 6, 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 以0.5的概率随机失活节点
            nn.Linear(128 * 6 * 6, 2048),  # 规定接下来的全连接层有2048个节点
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),  # 全连接层连接之前进行随机失活操作
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.method(x)
        x = torch.flatten(x, start_dim=1)  # 深度、高度、宽度展平，不改变batch
        return self.classifier(x)

    def _initialize_weights(self):  # 初始化权重的方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 如果m是卷积层的话，初始化它的权重和偏值
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


'''
class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Bottleneck(96, 32, 128),

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Bottleneck(256, 64, 512),

            nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            Bottleneck(768, 128, 768),

            nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            Bottleneck(1024, 128, 1280),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class Net(nn.Module):
    def __init__(self):
        super( Net, self ).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
'''


model = Net( num_classes = 5, init_weights = True )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pth_path = 'D:\[ Worksheet4 ] QG\QG_summer_camp\Daily/7.7-7.11\path\AlexNet_best.pth'
example = torch.randn( 1, 3, 224, 224 ).to(device)




model.load_state_dict(torch.load(pth_path))
model = model.to(device)
model.eval()


torch.onnx.export(model, example, "../data/Alex_flower.onnx" )
model_onnx = onnx.load( "../data/Alex_flower.onnx" )
onnx.checker.check_model(model_onnx)
print(onnx.helper.printable_graph(model_onnx.graph))

