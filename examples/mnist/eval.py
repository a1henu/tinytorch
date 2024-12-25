from __future__ import annotations

def evaluate(model, test_loader, loss_fn):
    total = 0
    correct = 0
    total_loss = 0

    for data, target in test_loader:
        output = model(data)
        loss = loss_fn(output, target)
        total_loss += loss.to_numpy()[0]

        pred_labels = output.to_numpy().argmax(axis=1)
        correct += (pred_labels == target.to_numpy()).sum()
        total += len(target)

    return correct / total, total_loss / len(test_loader)
    