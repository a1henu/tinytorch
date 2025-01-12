from __future__ import annotations
from tqdm import tqdm

def train(model, train_loader, loss_fn, epochs, optimizer_class, learning_rate):
    model.train()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.to_numpy()[0]
                
                pbar.set_postfix({'Loss': f"{total_loss / (i+1):.4f}"})
                pbar.update(1)
        
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")