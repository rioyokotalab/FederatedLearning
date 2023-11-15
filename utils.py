import math
import os
import random
import numpy as np
import torch
import torch.optim as optim

def set_random_seeds(seed_value=0, full_determinism=False):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    if full_determinism:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_optimizer(parameters, config):
    """Create an optimizer for training."""
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr, momentum=config.momentum,
                              dampening=config.dampening, weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
                               weight_decay=config.weight_decay, amsgrad=config.adam_amsgrad)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.lr, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps,
                                weight_decay=config.weight_decay, amsgrad=config.adam_amsgrad)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optimizer

def get_scheduler(optimizer, config):
    """Create a learning rate scheduler."""
    if config.lr_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.lr_scheduler == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    elif config.lr_scheduler == "linear":
        num_training_steps = config.communication_rounds * config.inner_loop
        num_warmup_steps = math.ceil(num_training_steps * config.warmup_ratio) if config.warmup_steps is None else config.warmup_steps
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    return scheduler

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret

def atleast_1d(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result
