import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from model.img2text import get_text_features, get_loss_img2text
from model.make_model_vi import load_clip_to_cpu


def do_train_stage3(cfg,
                    model,
                    img2text,
                    center_criterion,
                    train_loader_stage2_all,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    local_rank):
    log_period = cfg.SOLVER.STAGE3.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE3.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE3.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE3.MAX_EPOCHS

    logger = logging.getLogger("transreid_VI.train")
    logger.info('start training stage3')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    clip_model = load_clip_to_cpu(model.model_name, model.h_resolution, model.w_resolution, model.vision_stride_size)
    clip_model.to("cuda")

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    batch = cfg.SOLVER.STAGE3.IMS_PER_BATCH

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        # evaluator.reset()

        scheduler.step()

        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2_all):
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view)
                loss = get_loss_img2text(model, img2text, image_features, loss_img, loss_txt)
                pass




