from tinytorch import Tensor, nn
from tinytorch.data import DataLoader, MNIST
from tinytorch.optim import Adam

import matplotlib.pyplot as plt

class Diffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Diffusion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape([x.shape[0], 1, 28, 28])
        return x
    
def noise_forward(x, time_step=1, noise_level=0.1):
    noise = Tensor.randn(x.shape) * noise_level
    return x + noise * time_step

def train(dataloader, epochs, optimizer_class, learning_rate):
    model = Diffusion(784, 128, 32)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = noise_forward(images)
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.to_numpy()[0]
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Step [{i+1}/{len(dataloader)}], "
                      f"Loss: {total_loss / (i+1):.4f}")
    
    return model

def generate_images(model, n=10):
    images = model(Tensor.randn([n, 28, 28]) * 0.1)
    return images

if __name__ == "__main__":
    train_loader = DataLoader(MNIST(root="./data", train=True, download=True), batch_size=64)
    model = train(train_loader, 10, Adam, 3e-4)
    images = generate_images(model)
    plt.imsave("generated_images.png", images.to_numpy()[0].reshape(28, 28), cmap="gray")