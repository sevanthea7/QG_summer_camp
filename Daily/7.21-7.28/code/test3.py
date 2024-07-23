import mindcv
dataset = mindcv.create_dataset('cifar10', download=True)
network = mindcv.create_model('resnet50', pretrained=True)