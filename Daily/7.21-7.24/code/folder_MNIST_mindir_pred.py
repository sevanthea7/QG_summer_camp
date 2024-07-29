import mindspore
from mindspore import nn, context, Tensor
from mindspore.dataset import vision, transforms
import numpy as np
from PIL import Image
import os


def datapipe( dataset, batch_size ):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset



context.set_context( mode = mindspore.GRAPH_MODE )

graph = mindspore.load( "model2.mindir" )
model = nn.GraphCell( graph )


model.set_train(False)


def transform_image( img_path ):
    if isinstance( img_path, str):
        image = Image.open( img_path ).convert('L')

    transform = transforms.Compose([
        vision.Resize((28, 28)),
        vision.Normalize(mean=[0.5*255], std=[0.5*255]),
        vision.HWC2CHW()
    ])
    image = transform(image)
    image = np.expand_dims(image, axis=0)

    image = Tensor( image, mindspore.float32 )
    return image

def get_prediction( img_path ):
    tensor = transform_image( img_path )
    outputs = model( tensor )
    predicted = outputs.argmax( axis=1 ).asnumpy().item()
    return predicted


predictions = []
test_folder_path = '../data/MNIST_test_image'
test_names = os.listdir( test_folder_path )
for test_name in test_names:
    img_path = os.path.join( test_folder_path, test_name )
    # print( img_path )
    pred = get_prediction( str(img_path) )
    predictions.append( pred )

print( predictions )
