from tinytorch import Tensor, DeviceType
from tinytorch.data import MNIST, DataLoader
from tinytorch.funcs import (
    fc_forward, fc_backward, 
    conv2d_forward, conv2d_backward,
    relu_forward, relu_backward, 
    softmax_forward, 
    cross_entropy_forward, cross_entropy_backward
)

import numpy as np
from typing import Tuple, List

class Net:
    def __init__(self):
        # Conv1: 1x28x28 -> 6x24x24
        self.conv1_w = Tensor.randn([6, 1, 5, 5], DeviceType.CPU) * 0.1
        self.conv1_b = Tensor.from_numpy(np.zeros(6), DeviceType.CPU)
        
        # Conv2: 6x24x24 -> 16x20x20 
        self.conv2_w = Tensor.randn([16, 6, 5, 5], DeviceType.CPU) * 0.1
        self.conv2_b = Tensor.from_numpy(np.zeros(16), DeviceType.CPU)
        
        # FC: 16x20x20 -> 10
        self.fc_w = Tensor.randn([16 * 20 * 20, 10], DeviceType.CPU) * 0.1
        self.fc_b = Tensor.from_numpy(np.zeros([1, 10]), DeviceType.CPU)
        
        self.lr = 0.01
        
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass
        
        Args:
            x: Input tensor (batch_size, 1, 28, 28)
            
        Returns:
            output: Network output
            cache: Intermediate results for backprop
        """
        # First conv block
        conv1 = conv2d_forward(x, self.conv1_w, self.conv1_b, (0, 0), (1, 1))
        relu1 = relu_forward(conv1)
        
        # Second conv block
        conv2 = conv2d_forward(relu1, self.conv2_w, self.conv2_b, (0, 0), (1, 1))
        relu2 = relu_forward(conv2)
        
        # Flatten
        batch_size = relu2.shape()[0]
        flatten = relu2.reshape([batch_size, -1])
        
        # Fully connected
        fc = fc_forward(flatten, self.fc_w, self.fc_b)
        
        # Softmax
        output = softmax_forward(fc)
        
        # Cache for backprop (including pre-softmax fc)
        cache = [x, conv1, relu1, conv2, relu2, flatten, fc]
        return output, cache
    
    def backward(
        self, 
        pred: Tensor,
        target: Tensor, 
        cache: List[Tensor]
    ) -> float:
        """Backward pass
        
        Args:
            pred: Network predictions
            target: Target labels
            cache: Cached results from forward pass
            
        Returns:
            loss: Loss value
        """
        # Unpack cached tensors
        x, conv1, relu1, conv2, relu2, flatten, fc = cache
        
        # Compute loss using softmax output
        loss = cross_entropy_forward(fc, target)
        # Compute gradient using pre-softmax values
        grad = cross_entropy_backward(fc, target)
        
        # Backprop through FC
        grad_fc_x, grad_fc_w, grad_fc_b = fc_backward(
            flatten, self.fc_w, self.fc_b, fc, grad
        )
        
        # Reshape gradient to match conv output
        grad_flatten = grad_fc_x.reshape(relu2.shape())
        
        # Backprop through second conv block
        grad_relu2 = relu_backward(conv2, grad_flatten)
        grad_conv2_x, grad_conv2_w, grad_conv2_b = conv2d_backward(
            relu1, self.conv2_w, grad_relu2, (0, 0), (1, 1)
        )
        
        # Backprop through first conv block
        grad_relu1 = relu_backward(conv1, grad_conv2_x)
        grad_conv1_x, grad_conv1_w, grad_conv1_b = conv2d_backward(
            x, self.conv1_w, grad_relu1, (0, 0), (1, 1)
        )
        
        # Update weights
        self.conv1_w -= self.lr * grad_conv1_w
        self.conv1_b -= self.lr * grad_conv1_b
        self.conv2_w -= self.lr * grad_conv2_w
        self.conv2_b -= self.lr * grad_conv2_b
        self.fc_w -= self.lr * grad_fc_w
        self.fc_b -= self.lr * grad_fc_b
        
        return loss.to_numpy()[0]

def evaluate(model: Net, dataloader: DataLoader) -> Tuple[float, float]:
    """Evaluate model performance
    
    Args:
        model: Neural network model
        dataloader: Data loader
        
    Returns:
        accuracy: Classification accuracy
        avg_loss: Average loss
    """
    total = 0
    correct = 0
    total_loss = 0
    
    for images, labels in dataloader:
        # Forward pass
        pred, cache = model.forward(images)
        
        # Compute loss
        loss = cross_entropy_forward(pred, labels)
        total_loss += loss.to_numpy()[0]
        
        # Compute accuracy
        pred_labels = pred.to_numpy().argmax(axis=1)
        correct += (pred_labels == labels.to_numpy()).sum()
        total += len(labels)
    
    return correct / total, total_loss / len(dataloader)

def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5
) -> Net:
    """Train the network
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of epochs to train
        
    Returns:
        model: Trained model
    """
    # Create model
    model = Net()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Forward & backward pass
            pred, cache = model.forward(images)
            loss = model.backward(pred, labels, cache)
            total_loss += loss
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {total_loss / (i+1):.4f}")
        
        # Evaluate
        train_acc, train_loss = evaluate(model, train_loader)
        test_acc, test_loss = evaluate(model, test_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
              f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    return model

if __name__ == "__main__":
    """Train and test the CNN on MNIST dataset"""
    # Test data loading
    print("Loading datasets...")
    train_dataset = MNIST(root="./data", train=True, download=True)
    test_dataset = MNIST(root="./data", train=False, download=True)
    
    assert len(train_dataset) == 60000, "Wrong training set size"
    assert len(test_dataset) == 10000, "Wrong test set size"
    
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train model
    print("Training model...")
    model = train(train_loader, test_loader, epochs=1)
    
    # Evaluate model
    print("Evaluating model...")
    train_acc, train_loss = evaluate(model, train_loader)
    test_acc, test_loss = evaluate(model, test_loader)
    
    print("\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    assert train_acc > 0.8, "Training accuracy too low"
    assert test_acc > 0.8, "Test accuracy too low"


