from __future__ import annotations

def train(model, train_loader, loss_fn, epochs, optimizer_class, learning_rate, **optimizer_params):
    optimizer = optimizer_class(model.parameters(), lr=learning_rate, **optimizer_params)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.to_numpy()[0]
            
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Train Epoch: [{epoch + 1}/{epochs}], ",
                    f"Step: [{batch_idx + 1}/{len(train_loader)}], ",
                    f"Loss: {total_loss / (batch_idx + 1):.4f}"
                )