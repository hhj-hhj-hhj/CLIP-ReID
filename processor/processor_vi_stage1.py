import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1_rgb,
                    train_loader_stage1_ir,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1_rgb):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        labels_list_rgb = torch.stack(labels, dim=0).cuda()  # N
        image_features_list_rgb = torch.stack(image_features, dim=0).cuda()
    image_features.clear()
    labels.clear()
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1_ir):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        labels_list_ir = torch.stack(labels, dim=0).cuda()  # N
        image_features_list_ir = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH

        num_image_rgb = labels_list_rgb.shape[0]
        num_image_ir = labels_list_ir.shape[0]

        i_ter_rgb = num_image_rgb // batch
        i_ter_ir = num_image_ir // batch

    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list_rgb = torch.randperm(num_image_rgb).to(device)
        for i in range(i_ter_rgb + 1):
            optimizer.zero_grad()
            if i != i_ter_rgb:
                b_list_rgb = iter_list_rgb[i * batch:(i + 1) * batch]
            else:
                b_list_rgb = iter_list_rgb[i * batch:num_image_rgb]

            target = labels_list_rgb[b_list_rgb]
            image_features = image_features_list_rgb[b_list_rgb]
            with amp.autocast(enabled=True):
                text_features_rgb = model(label=target,img_modal = 1, get_text=True)

            loss_i2t = xent(image_features, text_features_rgb, target, target)
            loss_t2i = xent(text_features_rgb, image_features, target, target)

            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1_rgb),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
