
import torch.onnx
import re
import numpy as np
from LSTM_model import MyModel


def tokenlize(content):
    content = re.sub(r"([.!?])", r" \1", content)
    content = re.sub(r"[^a-zA-Z.!?]+", r" ", content)
    token = [i.strip().lower() for i in content.split()]
    return token


class Transform():
    def __init__(self):
        self.PAD_TAG = "PAD"
        self.UNK = 0

    def trans(self, sentence, ws, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [ws.get(word, self.UNK) for word in sentence]


temp_transform = Transform()

# Load vocabulary
ws = np.load("../data/aclImdb_trans.npy", allow_pickle=True).item()


text_input = "hello world"
tokens = tokenlize(text_input)
indices = temp_transform.trans(sentence=tokens, ws=ws, max_len=200)
# Convert indices to a tensor
indices_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1)  # (batch_size, seq_length)


model = MyModel()
load_path = '../data/aclImdb_model.pkl'
state_dict = torch.load(load_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()


input_tensor = indices_tensor


onnx_path = "../data/new_LSTSM_model.onnx"

# Export the model to ONNX format
torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 1: 'seq_length'}, 'output': {0: 'batch_size'}},
    opset_version=11,
    verbose=True
)

print(f"ONNX model successfully exported to {onnx_path}")
