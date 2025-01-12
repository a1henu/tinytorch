import os
import numpy as np
from numpy.typing import NDArray
import urllib.request
import zipfile
from typing import List, Tuple, Union
from PIL import Image
from tinytorch.data import DataSet

class TinyImageNet(DataSet):
    """Tiny ImageNet dataset"""
    
    URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    def __init__(
        self, 
        root: str = "./data", 
        train: bool = True, 
        download: bool = False
    ) -> None:
        """Initialize the Tiny ImageNet dataset
        
        Parameters:
            root: The root directory to store the dataset
            train: If True, load the training dataset, otherwise load the validation dataset
            download: If True, download the dataset from the internet
        """
        super().__init__()
        self.root = os.path.join(root, "TinyImageNet")
        self.train = train
        self.download = download
        
        if self.download:
            self._download()
            
        self.data_dir = os.path.join(self.root, 'tiny-imagenet-200')
        
        # Create class to index mapping
        self.class_to_idx = self._create_class_to_idx()
        
        if self.train:
            # Load training data
            self.images = []
            self.labels = []
            train_dir = os.path.join(self.data_dir, 'train')
            for class_name in os.listdir(train_dir):
                class_dir = os.path.join(train_dir, class_name, 'images')
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
                    img = img.resize((64, 64))  # Ensure consistent size
                    img = np.array(img)
                    self.images.append(img)
                    self.labels.append(self.class_to_idx[class_name])
            self.images = np.array(self.images)
            self.labels = np.array(self.labels, dtype=np.float64)
        else:
            # Load validation data
            val_dir = os.path.join(self.data_dir, 'val')
            with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
                self.images = []
                self.labels = []
                for line in lines:
                    img_name, class_name = line.split()[:2]
                    img_path = os.path.join(val_dir, 'images', img_name)
                    img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
                    img = img.resize((64, 64))  # Ensure consistent size
                    img = np.array(img)
                    self.images.append(img)
                    self.labels.append(self.class_to_idx[class_name])
                self.images = np.array(self.images)
                self.labels = np.array(self.labels, dtype=np.float64)
            
        # Reshape and normalize images
        self.images = self.images.reshape(-1, 3, 64, 64).astype(np.float64) / 255.0
            
    def _download(self) -> None:
        """Download the Tiny ImageNet dataset"""
        os.makedirs(self.root, exist_ok=True)
        
        # Download zip file
        filename = os.path.basename(self.URL)
        filepath = os.path.join(self.root, filename)
        
        if not os.path.exists(filepath):
            print(f'Downloading {self.URL}...')
            urllib.request.urlretrieve(self.URL, filepath)
            
            # Extract zip file
            print('Extracting...')
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.root)
                
    def _create_class_to_idx(self) -> Dict[str, int]:
        """Create a mapping from class names to indices"""
        train_dir = os.path.join(self.data_dir, 'train')
        class_names = sorted(os.listdir(train_dir))
        return {class_name: idx for idx, class_name in enumerate(class_names)}
    
    def __getitem__(
        self, 
        index: Union[int, slice, List, NDArray]
    ) -> Tuple[NDArray, NDArray]:
        """Get the item at the given index"""
        return self.images[index], self.labels[index]
    
    def __len__(self) -> int:
        """Get the length of the dataset"""
        return len(self.images)