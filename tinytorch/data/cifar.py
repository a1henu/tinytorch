import os
import numpy as np
from numpy.typing import NDArray
import urllib.request
import tarfile
import pickle
from typing import List, Tuple, Union

from tinytorch.data import DataSet

class CIFAR10(DataSet):
    """CIFAR10 dataset"""
    
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    def __init__(
        self, 
        root: str = "./data", 
        train: bool = True, 
        download: bool = False
    ) -> None:
        """Initialize the CIFAR10 dataset
        
        Parameters:
            root: The root directory to store the dataset
            train: If True, load the training dataset, otherwise load the test dataset
            download: If True, download the dataset from the internet
        """
        super().__init__()
        self.root = os.path.join(root, "CIFAR10")
        self.train = train
        self.download = download
        
        if self.download:
            self._download()
            
        self.data_dir = os.path.join(self.root, 'cifar-10-batches-py')
        
        if self.train:
            # Load training data from data_batch_1 to data_batch_5
            self.images = []
            self.labels = []
            for i in range(1, 6):
                batch_path = os.path.join(self.data_dir, f'data_batch_{i}')
                images, labels = self._load_batch(batch_path)
                self.images.append(images)
                self.labels.append(labels)
            self.images = np.concatenate(self.images)
            self.labels = np.concatenate(self.labels)
        else:
            # Load test data from test_batch
            test_path = os.path.join(self.data_dir, 'test_batch')
            self.images, self.labels = self._load_batch(test_path)
            
        # Reshape and normalize images
        self.images = self.images.reshape(-1, 3, 32, 32).astype(np.float64) / 255.0
            
    def _download(self) -> None:
        """Download the CIFAR10 dataset"""
        os.makedirs(self.root, exist_ok=True)
        
        # Download tar file
        filename = os.path.basename(self.URL)
        filepath = os.path.join(self.root, filename)
        
        if not os.path.exists(filepath):
            print(f'Downloading {self.URL}...')
            urllib.request.urlretrieve(self.URL, filepath)
            
            # Extract tar file
            print('Extracting...')
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=self.root)
                
    def _load_batch(self, filepath: str) -> Tuple[NDArray, NDArray]:
        """Load a batch of data
        
        Args:
            filepath: Path to the batch file
            
        Returns:
            Tuple of (images, labels)
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            
        return data[b'data'], np.array(data[b'labels'])
    
    def __getitem__(
        self, 
        index: Union[int, slice, List, NDArray]
    ) -> Tuple[NDArray, NDArray]:
        """Get the item at the given index"""
        return self.images[index], self.labels[index]
    
    def __len__(self) -> int:
        """Get the length of the dataset"""
        return len(self.images)