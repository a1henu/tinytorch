from __future__ import annotations

def train(model, train_loader, loss_fn, epochs, optimizer_class, learning_rate):
    model.train()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.to_numpy()[0]
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {total_loss / (i+1):.4f}")