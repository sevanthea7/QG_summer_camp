"""
训练测试函数
"""
import torch
from LSTM_model import MyModel
from torch.optim import Adam
from LSTM_load_dataset import get_dataloader, tokenlize
import LSTM_hyper_parameters as hyper_parameters
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = Adam( model.parameters(), lr=hyper_parameters.learning_rate )

def train(epochs):
    train_loss_list = []
    train_acc_list  = []
    test_acc_list   = []
    train_data_loader = get_dataloader(train=hyper_parameters.train)
    test_data_loader  = get_dataloader(train=hyper_parameters.test)

    for epoch in tqdm(range(epochs)):
        for idx, (input, target) in tqdm(enumerate(train_data_loader),
                                         total=len(train_data_loader),
                                         ascii=True, desc="train"):
            input, target = input.to(device), target.to(device).long()    # 放到 gpu 上运行
            optimizer.zero_grad()                                  # 每次迭代前将上一次的梯度置零
            output = model(input)
            loss = F.nll_loss(output, target)
            loss.backward()                                        # 损失反向传播
            optimizer.step()                                       # 梯度更新

            train_loss_list.append(loss.cpu().item())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            train_acc_list.append(cur_acc.cpu().item())

            if idx % 100 == 0:                                     # 每运行 100 个 batch_size ,保存一下模型
                torch.save(model.state_dict(), "../data/aclImdb_model.pkl")
                torch.save(optimizer.state_dict(), "../data/aclImdb_oprimizer.pkl")

        for idx_t, (input_t, target_t) in tqdm(enumerate(test_data_loader)):
            input_t, target_t = input_t.to(device), target_t.to(device)
            with torch.no_grad():
                output_t = model(input_t)

                pred_t = output_t.max(dim=-1)[-1]  # tensor.max() ([values_list], [indices_list])
                cur_acc_t = pred_t.eq(target_t).float().mean()  # 返回的值是否与 target 相同
                test_acc_list.append(cur_acc_t.cpu().item())

        print("epoch: ", epoch, "train_loss: ", np.mean(train_loss_list), "train_acc: ", np.mean(train_acc_list),
              "test_acc", np.mean(test_acc_list) )


def test_():
    loss_list = []
    acc_list  = []
    model.load_state_dict(torch.load("../data/aclImdb_model.pkl"))
    optimizer.load_state_dict(torch.load("../data/aclImdb_oprimizer.pkl"))
    data_loader = get_dataloader(train=hyper_parameters.test)

    for idx, (input, target) in tqdm(enumerate(data_loader),
                                     total=len(data_loader),
                                     ascii=True, desc="test"):
        input, target = input.to(device), target.to(device).long()
        with torch.no_grad():
            output = model(input)

            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())

            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())

    print("total loss, acc: ", np.mean(loss_list), np.mean(acc_list))

'''
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

def predict_text(text):
    temp_transform = Transform()
    labels = ['negative', 'positive']
    index_to_label = {idx: label for idx, label in enumerate(labels)}
    ws = np.load("../data/aclImdb_trans.npy", allow_pickle=True).item()

    tokens = tokenlize(text)
    indices = temp_transform.trans(sentence=tokens, ws=ws, max_len=200)
    input_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class_index = output.argmax(dim=-1).item()
        predicted_label = index_to_label.get(predicted_class_index, "Unknown")

    print(f"Text: {text} | Predicted label: {predicted_label}")
'''


if __name__ == '__main__':
    # train(hyper_parameters.epoch)
    test_()
    # predict_text("It's a bad movie, i won't watch it again")
