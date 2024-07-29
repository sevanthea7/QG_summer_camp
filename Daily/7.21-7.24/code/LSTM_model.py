
import torch.nn as nn
import torch.nn.functional as F
import LSTM_hyper_parameters as hyper_parameters

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(hyper_parameters.max_features, hyper_parameters.embedding_size)
        self.lstm      = nn.LSTM(input_size=hyper_parameters.embedding_size,
                                 hidden_size=hyper_parameters.lstm_hidden_size,
                                 num_layers=hyper_parameters.lstm_num_layer,
                                 batch_first=True,
                                 bidirectional=hyper_parameters.bidriectional,
                                 dropout=hyper_parameters.lstm_dropout)
        self.lstm_out = nn.LSTM(input_size=hyper_parameters.lstm_hidden_size*2,
                                hidden_size=hyper_parameters.lstm_out_hidden,
                                num_layers=hyper_parameters.lstm_out_num,
                                batch_first=True,
                                bidirectional=hyper_parameters.lstm_bidriectional)
        self.fc = nn.Linear(hyper_parameters.lstm_out_hidden, hyper_parameters.output_class)

    def forward(self, input):
        x = self.embedding(input)
        x, (h_n, c_n) = self.lstm(x)
        x_out, (h_out_n, c_out_n) = self.lstm_out(x)
        output = h_out_n[0]
        out = self.fc(output)
        return F.log_softmax(out, dim=-1)




