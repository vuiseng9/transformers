import torch
from functools import partial
from collections import defaultdict, OrderedDict

def annotate_module_static_attr(top_module, family_name=None):
    # static attr: 
    # first_name, last_name, class_name, is_leaf_module, leaf_has_weight
    if family_name is None:
        family = top_module.__class__.__name__.lower() + "class_as_family_name"
    else:
        family = family_name

    for parent_name, parent_module in top_module.named_modules():
        # handle top level because children loop below operate one level below, top level module will be missed 
        if parent_name == "":
            parent_module.first_name = family
            parent_module.last_name = ""

        for child_name, child_module in parent_module.named_children():
            child_module.first_name = child_name
            if parent_name == "":
                # just to handle the period if we dont do this conditional loop
                child_module.last_name = f"{family}"
            else:
                child_module.last_name = f"{family}.{parent_name}"
            
        # Following applies to every module
        parent_module.leaf_module = False
        if len(list(parent_module.children())) == 0:
            parent_module.is_leaf_module = True
            parent_module.leaf_has_weight = False
            if len(list(parent_module.parameters())) > 0:
                parent_module.leaf_has_weight = True

        parent_module.class_name = parent_module.__class__.__name__
        parent_module.full_name = f"{parent_module.last_name}.{parent_module.first_name}" # must be put at last


def calc_sparsity(tensor, wt_shape=False):
    if isinstance(tensor, torch.Tensor):
        rate = 1-(tensor.count_nonzero()/tensor.numel())
        if wt_shape is True:
            return rate.item(), tuple(tensor.shape)
        return rate.item()
    else:
        raise ValueError("expect torch tensor")
    

def calc_sparsity_by_head(raw_tensor, wt_shape=False):
    if isinstance(raw_tensor, torch.Tensor):
        raw_shape = tuple(raw_tensor.shape)
        tensor = raw_tensor.reshape(raw_shape[0], raw_shape[1], -1)
        numel_by_head = tensor.shape[-1]
        rate = (1-tensor.count_nonzero(dim=-1)/numel_by_head).mean()

        if wt_shape is True:
            return rate.item(), tuple(raw_tensor.shape)
        return rate.item()
    else:
        raise ValueError("expect torch tensor")
    
def create_sparsity_analyzer_hook(target_module, registry):
    registry[target_module.full_name] = OrderedDict()
    registry[target_module.full_name]['w_sparsity'] = calc_sparsity(target_module.weight)
    registry[target_module.full_name]['w_shape'] = tuple(target_module.weight.shape)

    registry[target_module.full_name]['x_sparsity'] = []
    registry[target_module.full_name]['x_shape'] = []
    
    registry[target_module.full_name]['y_sparsity'] = []
    registry[target_module.full_name]['y_shape'] = []
    
    def post_hook(module, args, output, registry):
        input_sparsity = calc_sparsity(args[0])
        registry[module.full_name]['x_sparsity'].append(input_sparsity)
        registry[module.full_name]['x_shape'].append(tuple(args[0].shape))
        output_sparsity = calc_sparsity(output)
        registry[module.full_name]['y_sparsity'].append(output_sparsity)
        registry[module.full_name]['y_shape'].append(tuple(output.shape))
        # print(f"I:{input_sparsity:.4f}, W: {weight_sparsity:.4f}, O: {output_sparsity:.4f} | {module.full_name}")
    return partial(post_hook, registry=registry)