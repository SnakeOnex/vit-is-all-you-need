import torch

def get_params_str(m): return f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M"

def get_lr_scheduler(optim, warmup_steps, train_steps, min_lr):
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1, s / warmup_steps))
    cos_lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, train_steps, eta_min=min_lr)
    constant_lr = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1)
    return torch.optim.lr_scheduler.SequentialLR(optim, [warmup_sched, cos_lr_sched, constant_lr], [warmup_steps, train_steps])
