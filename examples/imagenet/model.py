from tinytorch import nn

class Model(nn.Module):
    def __init__(self, num_classes=200):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x