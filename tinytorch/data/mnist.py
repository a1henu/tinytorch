import os
import numpy as np
from numpy.typing import NDArray
import urllib.request
import gzip
from typing import List, Tuple, Union

from tinytorch.data import DataSet

class MNIST(DataSet):
    """MNIST dataset"""

    URLS = {
        "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "test_images" : "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels" : "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    }
    
    def __init__(
        self, 
        root: str = "./data", 
        train: bool = True, 
        download: bool = False
    ) -> None:
        """Initialize the MNIST dataset
        
        Parameters:
            root: The root directory to store the dataset
            train: If True, load the training dataset, otherwise load the test dataset
            download: If True, download the dataset from the internet
        """
        super().__init__()
        self.root = os.path.join(root, "MNIST")
        self.train = train
        self.download = download
        
        if self.download:
            self._download()
        
        if self.train:
            images_path = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")
        
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)
            
    def _download(self) -> None:
        """Download the MNIST dataset"""
        os.makedirs(self.root, exist_ok=True)
        for filename, url in self.URLS.items():
            filepath = os.path.join(self.root, os.path.basename(url))
            if not os.path.exists(filepath):
                print(f'Downloading {url}...')
                urllib.request.urlretrieve(url, filepath)
        
    def _load_images(self, filepath: str) -> NDArray:
        """Load the images from the given filepath"""
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 1, 28, 28).astype(np.float64) / 255.0
    
    def _load_labels(self, filepath: str) -> NDArray:
        """Load the labels from the given filepath"""
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    def __getitem__(
        self, 
        index: Union[int, slice, List, NDArray]
    ) -> Tuple[NDArray, NDArray]:
        """Get the item at the given index"""
        return self.images[index], self.labels[index]
    
    def __len__(self) -> int:
        """Get the length of the dataset"""
        return len(self.images)