# TinyTorch ImageNet Example

This example demonstrates how to train and evaluate a neural network model on the Stanford CS231n Tiny ImageNet dataset(http://cs231n.stanford.edu/tiny-imagenet-200.zip) using TinyTorch.

## Requirements

- Python 3.10+
- NumPy
- TinyTorch

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/a1henu/tinytorch.git
    cd tinytorch/examples/imagenet
    ```

2. Install `tinytorch`, please refer to the [installation guide](../../README.md#installation).

## Usage

### Training

To train the model, run the following command:
```bash
python main.py train --batch_size 64 --epochs 10 --learning_rate 3e-4 --device cpu
```

### Evaluation

To evaluate the model, run the following command:
```bash
python main.py eval --batch_size 64 --device cpu
```

## Arguments

- `--batch_size` (`-b`): Batch size for training or evaluation (default: 64)
- `--epochs` (`-e`): Number of epochs for training (default: 10)
- `--learning_rate` (`-lr`): Learning rate for training (default: 3e-4)
- `--device`: Device to use for training or evaluation (`cpu` or `gpu`, default: `cpu`)

## Results

After training, the final results will be printed, including the training accuracy. During evaluation, the test accuracy will be printed.

## License

This project is licensed under the MIT License.
