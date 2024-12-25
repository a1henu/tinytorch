import os

from tinytorch import DeviceType, nn
from tinytorch.optim import SGD, Adam

from model import Model
from train import train
from eval import evaluate
from utils import load_mnist, save_model

from argparse import ArgumentParser

parser = ArgumentParser(description='Training and Evaluation')
subparsers = parser.add_subparsers(dest='mode', help='Choose mode: train or eval')

train_parser = subparsers.add_parser('train', help='Training mode')
train_parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training')
train_parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
train_parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help='Learning rate for training')
train_parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to use for training')

eval_parser = subparsers.add_parser('eval', help='Evaluation mode')
eval_parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for evaluation')
eval_parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to use for evaluation')

args = parser.parse_args()

device = DeviceType.CPU if args.device == 'cpu' else DeviceType.GPU
train_loader, test_loader = load_mnist(args.batch_size, device)
model = Model()
criterion = nn.CrossEntropyLoss()

if device == DeviceType.GPU:
    model.to_gpu()

if args.mode == 'train':
    train(model, train_loader, criterion, args.epochs, Adam, args.learning_rate)
    save_model(model, 'checkpoint/model.npz')
elif args.mode == 'eval':
    if os.path.exists('checkpoint/model.npz'):
        model.load('checkpoint/model.npz')
    else:
        raise FileNotFoundError('Model checkpoint not found')
    train_acc, train_loss = evaluate(model, train_loader, criterion)
    test_acc, test_loss = evaluate(model, test_loader, criterion)

    print("\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
else:
    raise ValueError('Invalid mode')