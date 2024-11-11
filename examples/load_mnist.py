import matplotlib.pyplot as plt

from tinytorch.data import MNIST, DataLoader

if __name__ == "__main__":
    mnist = MNIST(root = "./data", train=True, download=True)
    train_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape(), labels.shape(), sep="\n")
        
        if (i == 4):
            break
        
