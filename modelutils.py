import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

DEV = 'cuda'

TORCH_SEED = 42

TORCH_SEEDS = [2197414221, 3983550199, 2290376441, 1544162936, 3859964368]

def torch_empty_cache():
    if DEV == 'mps':
        torch.mps.empty_cache()
    elif DEV == 'cuda':
        torch.cuda.empty_cache()

def matinverse_device():
    if DEV == 'mps':
        return 'cpu'
    return DEV

def torch_synchronize():
    if DEV == 'mps':
        torch.mps.synchronize()
    elif DEV == 'cuda':
        torch.cuda.synchronize()

def find_layers(module, layers=[layer.Conv2d, layer.Linear], name=''):
    res = {}
    for name1, child in module.named_children():
        if type(child) in layers:
            res[int(name1)] = child
    return res

def find_layers_string(module, layers=[layer.Conv2d, layer.Linear], name=''):
    res = {}
    for name1, child in module.named_children():
        if type(child) in layers:
            res[name1] = child
    return res

def find_linear_neuron_pairs(module, name=''):
    """
    Finds all consecutive pairs of Linear and Activation layers within a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to search.
        name (str): The prefix of the module name (used for nested modules).

    Returns:
        list: A list of tuples, where each tuple contains:
              - A tuple with the full name and the Linear layer.
              - A tuple with the full name and the Activation layer.
    """
    pairs = []
    named_children_list = list(module.named_children())
    activation_layers = (neuron.LIFNode, neuron.ParametricLIFNode)

    for i, (name1, child) in enumerate(named_children_list):
        current_full_name = f"{name}.{name1}" if name else name1

        # Check for Linear then Activation pair among direct children
        if i > 0:
            prev_name1, prev_child = named_children_list[i - 1]
            previous_full_name = f"{name}.{prev_name1}" if name else prev_name1
            if isinstance(prev_child, nn.Linear) and isinstance(child, activation_layers):
                pairs.append(((previous_full_name, prev_child), (current_full_name, child)))

        # Recursively search in children
        pairs.extend(find_linear_neuron_pairs(child, current_full_name))

    return pairs