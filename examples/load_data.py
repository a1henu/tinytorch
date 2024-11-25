import matplotlib.pyplot as plt

from tinytorch.data import MNIST, CIFAR10, DataLoader

def load_mnist():
    mnist = MNIST(root = "./data", train=True, download=True)
    train_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape(), labels.shape(), sep="\n")
        
        if (i == 4):
            break
        
def load_cifar():
    cifar = CIFAR10(root = "./data", train=True, download=True)
    train_loader = DataLoader(cifar, batch_size=32, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape(), labels.shape(), sep="\n")
        
        if (i == 4):
            break

if __name__ == "__main__":
    load_mnist()
    load_cifar()
    