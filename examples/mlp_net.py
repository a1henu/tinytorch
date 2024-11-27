from typing import Tuple

import numpy as np
from tinytorch import Tensor
from tinytorch.nn import Module, Linear, ReLU
from tinytorch.data import MNIST, DataLoader
from tinytorch.funcs import CrossEntropy
from tinytorch.optim import SGD

class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape([x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def evaluate(model: SimpleNN, dataloader: DataLoader) -> Tuple[float, float]:
    total = 0
    correct = 0
    total_loss = 0
    
    for images, labels in dataloader:
        output = model(images)
        loss = CrossEntropy(labels)(output)
        total_loss += loss.to_numpy()[0]
        pred_labels = output.to_numpy().argmax(axis=1)
        correct += (pred_labels == labels.to_numpy()).sum()
        total += len(labels)
    
    return correct / total, total_loss / len(dataloader)

def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.01,
    optimizer_class=SGD
) -> SimpleNN:
    model = SimpleNN()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            output = model(images)
            loss = CrossEntropy(labels)(output)
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
    
    assert len(train_dataset) == 60000, "Wrong training set size"
    assert len(test_dataset) == 10000, "Wrong test set size"
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Training model with SGD...")
    model = train(train_loader, test_loader, epochs=5, learning_rate=0.01, optimizer_class=SGD)
    
    print("Evaluating model...")
    train_acc, train_loss = evaluate(model, train_loader)
    test_acc, test_loss = evaluate(model, test_loader)
    
    print("\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")