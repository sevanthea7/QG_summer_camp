import numpy as np
from mindx.sdk import Tensor
from mindx.sdk import base
import os
from torchvision import transforms
from PIL import Image

base.mx_init()

model_path = "/root/MyFiles/Alex_leaf/model/model.om"
device_id = 0
img_size = 224

labels = {
    "0": "daisy",
    "1": "dandelion",
    "2": "roses",
    "3": "sunflowers",
    "4": "tulips"
}


def transform_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    mdx_tensor = Tensor(image.numpy().astype(np.float32))
    return mdx_tensor


def get_prediction( img_path ):
    tensor = transform_image( img_path )
    model = base.model( modelPath = model_path, deviceId = device_id )

    output = model.infer([tensor])[0]
    output.to_host()
    output = np.array(output)
    # print( output )
    output = output.flatten()
    predicted_index = np.argmax(output)
    return predicted_index


predictions = []
test_folder_path = '/root/MyFiles/Alex_leaf/data'
test_names = os.listdir(test_folder_path)

for test_name in test_names:
    img_path = os.path.join(test_folder_path, test_name)
    pred_index = get_prediction(img_path)
    pred_label = labels[int(pred_index)]
    predictions.append(pred_label)
    result = str(pred_index) + ' : ' + str(pred_label)
    print(result)
