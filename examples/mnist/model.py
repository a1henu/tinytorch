from __future__ import annotations
from tinytorch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.cnn(x)