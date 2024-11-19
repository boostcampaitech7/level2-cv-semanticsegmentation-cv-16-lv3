from torch.optim import lr_scheduler

# MultiStepLR
def multi_step_lr(optimizer, **scheduler_parameter):
    return lr_scheduler.MultiStepLR(optimizer, **scheduler_parameter)

# CosineAnnealingLR
def cosine_annealing_lr(optimizer, **scheduler_parameter):
    return lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_parameter)

# ReduceLROnPlateau
def reduce_lr_on_plateau(optimizer, **scheduler_parameter):
    return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_parameter)