from __future__ import annotations
from typing import List, Any, Dict, Union

from ..tensor import Tensor
from .. import funcs

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

