from torch.optim import Adam, SGD, AdamW

def get_optimizer(optimizer_name: str):
    if optimizer_name == 'Adam':
        return Adam
    elif optimizer_name == 'SGD':
        return SGD
    elif optimizer_name == 'AdamW':
        return AdamW
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')