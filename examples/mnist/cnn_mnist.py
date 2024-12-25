from typing import Tuple

import numpy as np
import time
import os

from tinytorch import Tensor, nn, DeviceType
from tinytorch.data import MNIST, DataLoader
from tinytorch.optim import SGD, Adam

class SimpleCNN(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
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

def evaluate(model: SimpleCNN, dataloader: DataLoader) -> Tuple[float, float]:
    total = 0
    correct = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for images, labels in dataloader:
        output = model(images)
        loss = criterion(output, labels)
        total_loss += loss.to_numpy()[0]
        pred_labels = output.to_numpy().argmax(axis=1)
        correct += (pred_labels == labels.to_numpy()).sum()
        total += len(labels)
    
    return correct / total, total_loss / len(dataloader)

def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.01,
    optimizer_class=SGD
) -> SimpleCNN:
    model = SimpleCNN()
    model.to_gpu()
    
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.to_numpy()[0]
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {total_loss / (i+1):.4f}")
        
        train_acc, train_loss = evaluate(model, train_loader)
        test_acc, test_loss = evaluate(model, test_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
              f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    return model

if __name__ == "__main__":
    print("Loading datasets...")
    train_dataset = MNIST(root="./data", train=True, download=True)
    test_dataset = MNIST(root="./data", train=False, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, device=DeviceType.GPU)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, device=DeviceType.GPU)
    
    print("Training model with Adam...")
    start_time = time.time()
    model = train(train_loader, test_loader, epochs=10, learning_rate=3e-4, optimizer_class=Adam)
    end_time = time.time()
    
    print(f"Training took {end_time - start_time:.2f} seconds")
    
    print("Evaluating model...")
    train_acc, train_loss = evaluate(model, train_loader)
    test_acc, test_loss = evaluate(model, test_loader)
    
    print("\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("Saving model...")
    if os.path.exists("mnist_cnn_model.npz"):
        os.remove("mnist_cnn_model.npz")
    model.save("mnist_cnn_model.npz")