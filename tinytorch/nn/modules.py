from __future__ import annotations
from typing import List, Any, Dict, Union
import numpy as np
import os

from ..tensor import Tensor

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, Module] = {}

    def forward(self, *inputs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *inputs: Any) -> Any:
        return self.forward(*inputs)

    def parameters(self) -> List[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_parameter(self, name: str, param: Tensor) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Module) -> None:
        self._modules[name] = module

    def __setattr__(self, name: str, value: Union[Tensor, Module]) -> None:
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._parameters:
            return self._parameters[name]
        if name in self._modules:
            return self._modules[name]
        return super().__getattr__(name)
    
    def to_gpu(self) -> None:
        """
        Move the module and its parameters to GPU.
        """
        for name, param in self._parameters.items():
            param.to_gpu()
        for name, module in self._modules.items():
            module.to_gpu()

    def to_cpu(self) -> None:
        """
        Move the module and its parameters to CPU.
        """
        for name, param in self._parameters.items():
            param.to_cpu()
        for name, module in self._modules.items():
            module.to_cpu()
            
    def save(self, filepath: str) -> None:
        """
        Save the module's parameters to a file.
        """
        data = {}
        for name, param in self._parameters.items():
            data[name] = param.to_numpy()
        for name, module in self._modules.items():
            module_data = module.save_to_dict()
            for sub_name, sub_param in module_data.items():
                data[f"{name}.{sub_name}"] = sub_param
        
        np.savez(filepath, **data)

    def save_to_dict(self) -> Dict[str, Any]:
        """
        Save the module's parameters to a dictionary.
        """
        data = {}
        for name, param in self._parameters.items():
            data[name] = param.to_numpy()
        for name, module in self._modules.items():
            module_data = module.save_to_dict()
            for sub_name, sub_param in module_data.items():
                data[f"{name}.{sub_name}"] = sub_param
        return data

    def load(self, filepath: str) -> None:
        """
        Load the module's parameters from a file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: '{filepath}'")
        
        data = np.load(filepath)
        self.load_from_dict(data)

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load the module's parameters from a dictionary.
        """
        for name, param in self._parameters.items():
            if name not in data:
                raise ValueError(f"Parameter '{name}' not found in saved data")
            param_data = data[name]
            if list(param_data.shape) != list(param.shape):
                raise ValueError(f"Shape mismatch for parameter '{name}': expected {param.shape}, got {param_data.shape}")
            param = Tensor.from_numpy(param_data)
        
        for name, module in self._modules.items():
            module_data = {k[len(name)+1:]: v for k, v in data.items() if k.startswith(f"{name}.")}
            module.load_from_dict(module_data)