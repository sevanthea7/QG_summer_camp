from flask import Flask, jsonify, request
from PIL import Image
import io
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net, Tensor
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import mindspore.ops as ops
import numpy as np
import mindspore
from mindspore import nn, Tensor, context



'''
class AlexNet(nn.Cell):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, pad_mode='valid')
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, pad_mode='valid')
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, pad_mode='valid')
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, pad_mode='valid')
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(256 * 6 * 6, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.fc3 = nn.Dense(4096, num_classes )

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.max_pool3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = AlexNet( num_classes = 5 )
checkpoint_path = '../data/model.ckpt'
param_dict = load_checkpoint(checkpoint_path)
load_param_into_net(model, param_dict)
'''


app = Flask( __name__ )


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    image = np.array(image)

    transform = transforms.Compose([
        vision.Resize((28, 28)),
        vision.Normalize(mean=[0.5*255], std=[0.5*255]),
        vision.HWC2CHW()
    ])
    image = transform(image)
    image = np.expand_dims(image, axis=0)

    image = Tensor( image, mindspore.float32 )
    return image




'''
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, y = ops.ArgMaxWithValue(axis=1)(outputs)
    return y.asnumpy().item()
'''

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    predicted = outputs.argmax(axis=1).asnumpy().item()
    return predicted


context.set_context( mode = mindspore.GRAPH_MODE )

graph = mindspore.load( "model3.mindir" )
model = nn.GraphCell(graph)




@app.route('/predict', methods=['POST'] )
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_bytes = image.read()
    prediction = get_prediction(image_bytes)
    return jsonify({'prediction': prediction}), 200


if __name__ == '__main__':
    app.run( debug = True )