import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor

def network():
    model = nn.SequentialCell(
                nn.Flatten(),
                nn.Dense(28*28, 512),
                nn.ReLU(),
                nn.Dense(512, 512),
                nn.ReLU(),
                nn.Dense(512, 10))
    return model


model = network()
mindspore.save_checkpoint(model, "../data/model.ckpt")


param_dict = mindspore.load_checkpoint("../data/model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)

model = network()
inputs = Tensor(np.ones([1, 1, 28, 28]).astype(np.float32))
mindspore.export(model, inputs, file_name="model", file_format="MINDIR")
mindspore.set_context(mode=mindspore.GRAPH_MODE)
graph = mindspore.load("model.mindir")
model = nn.GraphCell(graph)
outputs = model(inputs)
print(outputs.shape)