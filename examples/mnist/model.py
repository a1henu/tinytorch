from __future__ import annotations
from tinytorch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # (1, 28, 28) -conv-> (6, 24, 24) -pool-> (6, 12, 12)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        # (6, 12, 12) -> (16, 10, 10) -pool-> (16, 5, 5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 3))
        
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = x.reshape([x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x