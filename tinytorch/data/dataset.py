from numpy.typing import NDArray
from typing import List, Tuple, Union

class DataSet:
    """Abstract dataset class"""
    def __init__(self):
        """Initialize the dataset"""
        pass
    
    def __getitem__(
        self, 
        index: Union[int, slice, List, NDArray]
    ) -> Tuple[NDArray, NDArray]:
        """Get item from the dataset"""
        raise NotImplementedError

    def __len__(self) -> int:
        """Get the length of the dataset"""
        raise NotImplementedError
    