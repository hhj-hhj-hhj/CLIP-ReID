import torch
from torch import optim

def make_optimizer_1stage(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
    return optimizer



def make_optimizer_2stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue   
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center

def make_optimizer_3stage(args, img2text):
    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    named_parameters = list(img2text.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    return optimizer


def make_optimizer_4stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.STAGE4.BASE_LR
        weight_decay = cfg.SOLVER.STAGE4.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE4.BASE_LR * cfg.SOLVER.STAGE4.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE4.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE4.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.SOLVER.STAGE4.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE4.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE4.MOMENTUM)
    elif cfg.SOLVER.STAGE4.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE4.BASE_LR, weight_decay=cfg.SOLVER.STAGE4.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE4.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE4.CENTER_LR)

    return optimizer, optimizer_center