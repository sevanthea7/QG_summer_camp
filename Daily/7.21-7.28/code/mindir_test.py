import mindspore
from mindspore import nn, context, Tensor
from mindspore.dataset import vision, transforms
import numpy as np
from mindspore.dataset import MnistDataset
import mindspore.dataset as ds
import mindspore.ops as ops
import download as download

'''  
def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Resize((256, 256)),
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    for transform in image_transforms:
        dataset = dataset.map(operations=transform, input_columns='image')
    dataset = dataset.map(operations=label_transform, input_columns='label')
    dataset = dataset.batch(batch_size)

    return dataset

'''


class LSTM_Model(nn.Cell):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.fc = nn.Dense(hidden_dim, output_dim)

    def construct(self, x):
        h0 = Tensor(np.zeros((self.layer_dim, x.shape[0], self.hidden_dim)), mindspore.float32)
        c0 = Tensor(np.zeros((self.layer_dim, x.shape[0], self.hidden_dim)), mindspore.float32)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# 测试模型
input_dim = 10
hidden_dim = 20
layer_dim = 2
output_dim = 1

model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)


train_dataset = ds.ImageFolderDataset("../data/aclImdb/train" )
test_dataset = ds.ImageFolderDataset("../data/aclImdb/test" )


'''
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




model = AlexNet( num_classes = 5, init_weights = True )
'''

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 0.01 )


def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train( model, dataset ):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 10 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
            
def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)
    test(model, test_dataset, loss_fn)
print("Done!")


mindspore.save_checkpoint(model, "model3.ckpt")
print("Saved Model to model.ckpt")
