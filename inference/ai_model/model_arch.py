import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(224 * 224 * 3, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
