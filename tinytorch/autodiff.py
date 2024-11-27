from typing import List, Dict, Tuple
from functools import reduce
from operator import add

from .node import Node

def find_topo_sort(node_list: List[Node]) -> List[Node]:
    visited = set()
    topo_order = []
    for node in node_list:
        if node not in visited: 
            topo_sort_dfs(node, visited, topo_order)
    return topo_order
    

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    for input in node.inputs:
        topo_sort_dfs(input, visited, topo_order)
    visited.add(node)
    topo_order.append(node)
    

def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_tensor] = [out_grad]

    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        node.grad = reduce(add, node_to_output_grads_list[node])
        
        if node.is_leaf():
            continue
        
        for i, grad in enumerate(node.op.gradient_as_tuple(node.grad, node)):
            input = node.inputs[i]
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = []
            node_to_output_grads_list[input].append(grad)
            
        





