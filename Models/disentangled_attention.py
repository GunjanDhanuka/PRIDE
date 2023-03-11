from transformers import DebertaConfig, DebertaModel
import torch
import torch.nn as nn
from dltm import GRUModel, LSTMModel, RNNModel


class DisentangledAttention(nn.Module):
    """Disentangled Attention module"""

    def __init__(self, hidden_size=1024, num_hidden_layers=1, num_attention_heads=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        ## Updating the config dictionary
        config_dict = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
        }
        config = DebertaConfig(**config_dict)

        self.model = DebertaModel(config)

    def forward(self, x):
        out = self.model(inputs_embeds=x).last_hidden_state

        return out


class GRUDisentangledAttention(nn.Module):
    """A GRU layer on top of Disentangled Attention module"""

    def __init__(self, hidden_size=1024, num_hidden_layers=1, num_attention_heads=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        ## Updating the config dictionary
        config_dict = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
        }
        config = DebertaConfig(**config_dict)

        self.model = DebertaModel(config)
        self.gru = GRUModel(
            input_dim=self.hidden_size, hidden_dim=self.hidden_size // 2, num_layers=1
        )

    def forward(self, x):
        out = self.model(inputs_embeds=x).last_hidden_state
        out = self.gru(out)
        return out


class LSTMDisentangledAttention(nn.Module):
    """A LSTM layer on top of Disentangled Attention module"""

    def __init__(self, hidden_size=1024, num_hidden_layers=1, num_attention_heads=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        ## Updating the config dictionary
        config_dict = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
        }
        config = DebertaConfig(**config_dict)

        self.model = DebertaModel(config)
        self.lstm = LSTMModel(
            input_dim=self.hidden_size, hidden_dim=self.hidden_size // 2, num_layers=1
        )

    def forward(self, x):
        out = self.model(inputs_embeds=x).last_hidden_state
        out = self.lstm(out)
        return out


class RNNDisentangledAttention(nn.Module):
    """A RNN layer on top of Disentangled Attention module"""

    def __init__(self, hidden_size=1024, num_hidden_layers=1, num_attention_heads=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        ## Updating the config dictionary
        config_dict = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
        }
        config = DebertaConfig(**config_dict)

        self.model = DebertaModel(config)
        self.rnn = RNNModel(
            input_dim=self.hidden_size, hidden_dim=self.hidden_size // 2, num_layers=1
        )

    def forward(self, x):
        out = self.model(inputs_embeds=x).last_hidden_state
        out = self.rnn(out)
        return out


if __name__ == "__main__":
    x = torch.randn((30, 32, 1024))
    model = DisentangledAttention()
    for name, p in model.named_parameters():
        print(name, p.size())
