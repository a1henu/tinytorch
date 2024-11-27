from __future__ import annotations

class Ops:
    def __call__(self):
        raise NotImplementedError()
    
    def compute(self, *args):
        raise NotImplementedError()
    
    def gradient(self, out_grad, node):
        raise NotImplementedError()
    
    def gradient_as_tuple(self, out_grad, node):
        grads = self.gradient(out_grad, node)
        if isinstance(grads, tuple):
            return grads
        elif isinstance(grads, list):
            return tuple(grads)
        else:
            return (grads,)
        
        
