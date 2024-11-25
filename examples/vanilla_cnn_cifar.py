from tinytorch import Tensor, DeviceType
from tinytorch.data import CIFAR10, DataLoader
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
    def __init__(self, device: DeviceType = DeviceType.GPU):
        self.device = device
        
        # Conv1: 3x32x32 -> 32x28x28
        self.conv1_w = Tensor.randn([32, 3, 5, 5], device) * 0.1
        self.conv1_b = Tensor.zeros([32], device)
        
        # Conv2: 32x28x28 -> 64x24x24
        self.conv2_w = Tensor.randn([64, 32, 5, 5], device) * 0.1
        self.conv2_b = Tensor.zeros([64], device)
        
        # Conv3: 64x24x24 -> 128x20x20
        self.conv3_w = Tensor.randn([128, 64, 5, 5], device) * 0.1
        self.conv3_b = Tensor.zeros([128], device)
        
        # FC: 128x20x20 -> 10
        self.fc_w = Tensor.randn([128 * 20 * 20, 10], device) * 0.1
        self.fc_b = Tensor.zeros([1, 10], device)
        
        self.lr = 0.01
        
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 32, 32)
            
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
        
        # Third conv block
        conv3 = conv2d_forward(relu2, self.conv3_w, self.conv3_b, (0, 0), (1, 1))
        relu3 = relu_forward(conv3)
        
        # Flatten
        batch_size = relu3.shape()[0]
        flatten = relu3.reshape([batch_size, -1])
        
        # Fully connected
        fc = fc_forward(flatten, self.fc_w, self.fc_b)
        
        # Softmax
        output = softmax_forward(fc)
        
        # Cache for backprop
        cache = [x, conv1, relu1, conv2, relu2, conv3, relu3, flatten, fc]
        return output, cache
    
    def backward(
        self, 
        pred: Tensor,
        target: Tensor, 
        cache: List[Tensor]
    ) -> float:
        """Backward pass"""
        # Unpack cached tensors
        x, conv1, relu1, conv2, relu2, conv3, relu3, flatten, fc = cache
        
        # Compute loss and gradient
        loss = cross_entropy_forward(pred, target)
        grad = cross_entropy_backward(fc, target)
        
        # Backprop through FC
        grad_fc_x, grad_fc_w, grad_fc_b = fc_backward(
            flatten, self.fc_w, self.fc_b, fc, grad
        )
        
        # Reshape gradient
        grad_flatten = grad_fc_x.reshape(relu3.shape())
        
        # Backprop through third conv block
        grad_relu3 = relu_backward(conv3, grad_flatten)
        grad_conv3_x, grad_conv3_w, grad_conv3_b = conv2d_backward(
            relu2, self.conv3_w, grad_relu3, (0, 0), (1, 1)
        )
        
        # Backprop through second conv block
        grad_relu2 = relu_backward(conv2, grad_conv3_x)
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
        self.conv3_w -= self.lr * grad_conv3_w
        self.conv3_b -= self.lr * grad_conv3_b
        self.fc_w -= self.lr * grad_fc_w
        self.fc_b -= self.lr * grad_fc_b
        
        return loss.to_numpy()[0]

def evaluate(model: Net, dataloader: DataLoader) -> Tuple[float, float]:
    """Evaluate model performance"""
    total = 0
    correct = 0
    total_loss = 0
    
    for images, labels in dataloader:
        pred, cache = model.forward(images)
        loss = cross_entropy_forward(pred, labels)
        total_loss += loss.to_numpy()[0]
        
        pred_labels = pred.to_numpy().argmax(axis=1)
        correct += (pred_labels == labels.to_numpy()).sum()
        total += len(labels)
    
    return correct / total, total_loss / len(dataloader)

def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: DeviceType = DeviceType.GPU,
    epochs: int = 10
) -> Net:
    """Train the network"""
    model = Net(device)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Ensure data is on the correct device
            if images.device() != device:
                images.to_gpu() if device == DeviceType.GPU else images.to_cpu()
            if labels.device() != device:
                labels.to_gpu() if device == DeviceType.GPU else labels.to_cpu()
                
            pred, cache = model.forward(images)
            loss = model.backward(pred, labels, cache)
            total_loss += loss
            
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
    """Train and test the CNN on CIFAR10 dataset on GPU"""
    device = DeviceType.GPU 
    
    print("Loading datasets...")
    train_dataset = CIFAR10(root="./data", train=True, download=True)
    test_dataset = CIFAR10(root="./data", train=False, download=True)
    
    assert len(train_dataset) == 50000, "Wrong training set size"
    assert len(test_dataset) == 10000, "Wrong test set size"
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, device=device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, device=device)
    
    print(f"Training model on {device}...")
    model = train(train_loader, test_loader, device=device, epochs=10)
    
    print("Evaluating model...")
    train_acc, train_loss = evaluate(model, train_loader)
    test_acc, test_loss = evaluate(model, test_loader)
    
    print("\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    
    