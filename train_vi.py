from utils.logger import setup_logger
from datasets.make_dataloader_vi import make_dataloader
from model.make_model_vi import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage, make_optimizer_3stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from solver.scheduler_vi import cosine_lr
from loss.make_loss import make_loss
from processor.processor_vi_stage1 import do_train_stage1
from processor.processor_vi_stage2 import do_train_stage2
from processor.processor_vi_stage3 import do_train_stage3
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from model.img2text import IMG2TEXT

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print('开始加载配置文件')
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    # 第三阶段所需参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    # parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    # parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    # parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")

    # 结束第三阶段所需参数
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid_VI", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print('开始加载数据集')

    train_loader_stage2_all, train_loader_stage2_rgb, train_loader_stage2_ir, train_loader_stage1_all, train_loader_stage1_rgb, train_loader_stage1_ir, val_loader, num_query, num_classes_all, num_classes_rgb, num_classes_ir, camera_num_all, camera_num_rgb, camera_num_ir, view_num_all, view_num_rgb, view_num_ir = make_dataloader(
        cfg)

    print('数据集加载完成,开始构建模型')

    model = make_model(cfg, num_class=num_classes_all, camera_num=camera_num_all, view_num = view_num_all)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes_all)

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)

    print('开始一阶段的训练')

    do_train_stage1(
        cfg,
        model,
        train_loader_stage1_rgb,
        train_loader_stage1_ir,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2_rgb,
        train_loader_stage2_ir,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank
    )

    img2text = IMG2TEXT()
    optimizer_3stage = make_optimizer_3stage(args, img2text)
    scheduler_3stage = cosine_lr(optimizer_3stage, cfg.SOLVER.STAGE3.BASE_LR,
                                 cfg.SOLVER.STAGE3.WARMUP_EPOCHS * len(train_loader_stage2_all),
                                 cfg.SOLVER.STAGE3.MAX_EPOCHS * len(train_loader_stage2_all))

    do_train_stage3(
        cfg,
        model,
        img2text,
        center_criterion,
        train_loader_stage2_all,
        optimizer_3stage,
        # optimizer_center_3stage,
        scheduler_3stage,
        loss_func,
        args.local_rank
    )