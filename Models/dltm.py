import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        num_layers=2,
        batch_first=True,
        dropout=0.6,
        bidirectional=True,
    ):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        output, hn = self.gru(x)
        return output


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        num_layers=2,
        batch_first=True,
        dropout=0.6,
        bidirectional=True,
    ):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        return output


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        num_layers=2,
        batch_first=True,
        dropout=0.6,
        bidirectional=True,
    ):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        return output


class TConvModel(nn.Module):
    def __init__(self, input_dim=1024):
        super(TConvModel, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
        )

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv1d(out)
        return out.permute(0, 2, 1)
