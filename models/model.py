import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneEncoder(nn.Module):
    def __init__(self, layer_size, kernel_size, dilation):
        super().__init__()

        padding = ((kernel_size - 1) ** dilation) // 2

        self.conv1 = nn.Conv1d(
            in_channels=layer_size,
            out_channels=layer_size,
            kernel_size=kernel_size,
            padding=padding,
            dilaation=dilation,
            groups=layer_size
        )

        self.activation = nn.silu()

    def forward(self, x):

        x = self.conv1(x)

        x = self.activation(x)

        return x


class GeneModel(nn.Module):
    def __init__(self, layer_size=128, n_layers=64, n_embeddings=6):
        super().__init__()

        self.embeddings = nn.Embeddings(
            num_embeddings=num_embeddings,
            embedding_dim=layer_size,
            padding_idx=0
        )

        kernel_size = 5
        dilation = 2


        self.encoders = nn.ModuleList([
            GeneEncoder(layer_size=layer_size, kernel_size=kernel_size, dilation=dilation)
        ])

        self.pred_head = nn.Linear(layer_size, 1)

    def forward(self, x):

        x = self.encoders(x)

        x = self.pred_head(x)

        return x
