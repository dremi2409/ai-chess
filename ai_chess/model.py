import torch
import torch.nn as nn

class SimpleChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.sigmoid(self.out(x))
