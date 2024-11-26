from __future__ import annotations
from typing import Iterator, Tuple

import numpy as np

from tinytorch.data import DataSet
from tinytorch import Tensor, DeviceType

class DataLoader:
    """Generic data loader class for batched data loading"""
    def __init__(
        self, 
        dataset: DataSet, 
        batch_size: int = 32, 
        shuffle: bool = True,
        device: DeviceType = DeviceType.CPU
    ) -> None:
        """Initialize the data loader
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data at each epoch
            device: Device to load the tensors to
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.indices = np.arange(len(dataset))
        
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Create new iterator
        
        Returns:
            Iterator yielding (inputs, targets) tuples of tensors
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        for start_idx in range(0, len(self.dataset), self.batch_size):
            # get data of the batch
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            inputs, targets = self.dataset[batch_indices]
            
            yield Tensor.from_numpy(inputs, self.device), Tensor.from_numpy(targets, self.device)
            
    def __len__(self) -> int:
        """Get the number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size