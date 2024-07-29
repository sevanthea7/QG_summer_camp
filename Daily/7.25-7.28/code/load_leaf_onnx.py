import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image
import os


onnx_path = "../data/Alex_leaf.onnx"
session = ort.InferenceSession( onnx_path )


def transform_image( img_path ):
    image = Image.open( img_path ).convert('RGB')
    transform = transforms.Compose( [
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ] )
    image = transform( image )
    image = image.unsqueeze( 0 )
    return image

def get_prediction( img_path ):
    tensor = transform_image( img_path )
    ort_inputs = {session.get_inputs()[0].name: tensor.numpy()}
    ort_outs = session.run(None, ort_inputs)
    outputs = ort_outs[0]
    print( outputs )
    predicted = np.argmax(outputs, axis=1).item()
    return predicted

label = {
  "0": "Cassava Bacterial Blight (CBB)",
  "1": "Cassava Brown Streak Disease (CBSD)",
  "2": "Cassava Green Mottle (CGM)",
  "3": "Cassava Mosaic Disease (CMD)",
  "4": "Healthy"
}


predictions = []
test_folder_path = '../data/test_leaf'
test_names = os.listdir( test_folder_path )
for test_name in test_names:
    img_path = os.path.join( test_folder_path, test_name )
    pred = label[str(get_prediction( str(img_path) ))]
    predictions.append( pred )

print( predictions )

