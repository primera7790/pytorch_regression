import torch
import torch.nn as nn


class ModelReg(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_count):
        super().__init__()

        self.input_size = input_size
        self.layers = nn.ModuleList()

        for c in range(hidden_count):
            self.layers.add_module(f'linear_{c}', nn.Linear(input_size, hidden_size))
            self.layers.add_module(f'act_{c}', nn.ReLU())

            input_size = hidden_size

        self.layers.add_module('linear_last', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def test(self, batch_size, device):
        input_test_data = torch.rand([batch_size, self.input_size], dtype=torch.float32).to(device)
        print(f'Input shape: {input_test_data.shape}')
        out = self(input_test_data)
        print(f'Output shape: {out.shape}')