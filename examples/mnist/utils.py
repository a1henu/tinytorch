from __future__ import annotations
import os

from tinytorch.data import MNIST, DataLoader

def load_mnist(batch_size, device):
    train_data = MNIST(root="./data", download=True, train=True)
    test_data = MNIST(root="./data", download=True, train=False)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, device=device)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, device=device)
    
    return train_loader, test_loader

def save_model(model, file_path):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    model.save(file_path)