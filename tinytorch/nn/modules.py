from __future__ import annotations
from typing import List, Any, Dict, Union
import numpy as np
import os

from ..tensor import Tensor

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, Module] = {}
        self._device: str = 'cpu'
        
        self.training: bool = True
    
    def train(self) -> None:
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self) -> None:
        self.training = False
        for module in self._modules.values():
            module.eval()

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
        self._device = 'gpu'
        for name, param in self._parameters.items():
            param.to_gpu()
        for name, module in self._modules.items():
            module.to_gpu()

    def to_cpu(self) -> None:
        """
        Move the module and its parameters to CPU.
        """
        self._device = 'cpu'
        for name, param in self._parameters.items():
            param.to_cpu()
        for name, module in self._modules.items():
            module.to_cpu()

    def save(self, save_path: str) -> None:
        """
        Save all parameters of the module to a single file.

        Parameters:
            save_path (str): Path to save the parameters to.
        """
        save_data = {}

        # Save parameters
        for name, param in self._parameters.items():
            save_data[f"parameters/{name}"] = param.to_numpy()

        # Recursively save sub-modules
        def save_submodule(sub_module: Module, prefix: str):
            for name, param in sub_module._parameters.items():
                save_data[f"{prefix}parameters/{name}"] = param.to_numpy()
            for name, sub_sub_module in sub_module._modules.items():
                save_submodule(sub_sub_module, f"{prefix}{name}/")

        for name, sub_module in self._modules.items():
            save_submodule(sub_module, f"modules/{name}/")

        # Save to file
        np.savez_compressed(save_path, **save_data)

    def load(self, load_path: str) -> None:
        """
        Load all parameters into the module from a single file.

        Parameters:
            load_path (str): Path to load the parameters from.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"The file {load_path} does not exist.")

        # Load data
        load_data = np.load(load_path)

        # Load parameters
        for name in self._parameters.keys():
            param_key = f"parameters/{name}"
            if param_key in load_data:
                self._parameters[name] = Tensor.from_numpy(load_data[param_key])

        # Recursively load sub-modules
        def load_submodule(sub_module: Module, prefix: str):
            for name in sub_module._parameters.keys():
                param_key = f"{prefix}parameters/{name}"
                if param_key in load_data:
                    sub_module._parameters[name] = Tensor.from_numpy(load_data[param_key])
            for name, sub_sub_module in sub_module._modules.items():
                load_submodule(sub_sub_module, f"{prefix}{name}/")

        for name, sub_module in self._modules.items():
            load_submodule(sub_module, f"modules/{name}/")   
        
        if self._device == 'gpu':
            self.to_gpu()
        elif self._device == 'cpu':
            self.to_cpu()
        else:
            raise ValueError(f"Invalid device type: {self._device}")
