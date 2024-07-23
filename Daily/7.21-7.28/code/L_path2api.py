from flask import Flask, jsonify, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import io



app = Flask( __name__ )



class AlexNet(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(AlexNet, self).__init__()
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
        x = torch.flatten( x, start_dim=1 )
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


model = AlexNet( num_classes = 5, init_weights = True )
model.load_state_dict(torch.load('../data/model.pth'))
model.eval()



def transform_image( image_bytes ):
    transform = transforms.Compose( [
        transforms.Resize( ( 224, 224 ) ),
        transforms.ToTensor(),
        transforms.Normalize( ( 0.5, 0.5, 0.5 ), ( 0.5, 0.5, 0.5 ) )
    ] )

    image = Image.open( io.BytesIO( image_bytes ) ).convert( 'RGB' )
    return transform( image ).unsqueeze( 0 )



def get_prediction( image_bytes ):
    tensor = transform_image( image_bytes )
    outputs = model.forward( tensor )
    _, y = outputs.max( 1 )
    return y.item()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_bytes = image.read()
    prediction = get_prediction(image_bytes)
    with open('../data/classes.json', 'r') as f:
        class_i = json.load(f)
    return jsonify({'prediction': class_i[ str(prediction) ] } ), 200


if __name__ == '__main__':
    app.run( debug=True )