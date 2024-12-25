from tinytorch import DeviceType, nn
from tinytorch.optim import SGD, Adam

from model import Model
from train import train
from eval import eval
from utils import load_mnist

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