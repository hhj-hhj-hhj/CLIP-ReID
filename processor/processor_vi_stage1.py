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

        # i_ter_rgb = num_image_rgb // batch
        # i_ter_ir = num_image_ir // batch

    del labels, image_features
    # 相机ID对应的模态，3号和6号相机是红外即模态0，其他相机是可见光即模态1
    # cam2modal = {
    #     1 : 1,
    #     2 : 1,
    #     3 : 0,
    #     4 : 1,
    #     5 : 1,
    #     6 : 0
    # }
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        # 将两种模态的图片拓展到一样多的数量
        if num_image_rgb > num_image_ir:
            iter_list_rgb = torch.randperm(num_image_rgb).to(device)
            iter_list_ir = torch.cat([torch.randperm(num_image_ir), torch.randint(0, num_image_ir, (num_image_rgb - num_image_ir,))],
                                     dim=0).to(device)
        elif num_image_rgb == num_image_ir:
            iter_list_rgb = torch.randperm(num_image_rgb).to(device)
            iter_list_ir = torch.randperm(num_image_ir).to(device)
        else:
            iter_list_ir = torch.randperm(num_image_ir).to(device)
            iter_list_rgb = torch.cat([torch.randperm(num_image_rgb), torch.randint(0, num_image_rgb, (num_image_ir - num_image_rgb,))],
                                      dim=0).to(device)
            
        i_ter = len(iter_list_rgb) // batch
        
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list_rgb = iter_list_rgb[i * batch:(i + 1) * batch]
                b_list_ir = iter_list_ir[i * batch:(i + 1) * batch]
            else:
                b_list_rgb = iter_list_rgb[i * batch:len(iter_list_rgb)]
                b_list_ir = iter_list_ir[i * batch:len(iter_list_rgb)]


            target_rgb = labels_list_rgb[b_list_rgb]
            target_ir = labels_list_ir[b_list_ir]


            image_features_rgb = image_features_list_rgb[b_list_rgb]
            image_features_ir = image_features_list_ir[b_list_ir]

            # cam_modal = torch.tensor([cam2modal[int(i)] for i in cam], device=device)

            with amp.autocast(enabled=True):
                text_features_rgb = model(label=target_rgb,img_modal = 1, get_text=True)
                text_features_ir = model(label=target_ir,img_modal = 0, get_text=True)


            loss_i2t_rgb = xent(image_features_rgb, text_features_rgb, target_rgb, target_rgb)
            loss_t2i_rgb = xent(text_features_rgb, image_features_rgb, target_rgb, target_rgb)

            loss_i2t_ir = xent(image_features_ir, text_features_ir, target_ir, target_ir)
            loss_t2i_ir = xent(text_features_ir, image_features_ir, target_ir, target_ir)


            loss = loss_i2t_rgb + loss_t2i_rgb + loss_i2t_ir + loss_t2i_ir

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), i_ter + 1,
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:

            # 测试开始
            print(f'Test Epoch : {epoch}')




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
