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
from datasets.make_dataloader_all import make_dataloader_all


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    print('开始加载配置文件')
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )

    # 第三阶段所需参数
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    print(cfg)

    print('开始加载数据集')


    # optimizer_1stage = make_optimizer_1stage(cfg, model)
    # scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
    #                     warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)


    train_sp_loader = make_dataloader_all(cfg)

    for i, batch in enumerate(train_sp_loader):
        imgs_rgb, imgs_ir, labels_rgb, labels_ir = batch
        print(len(imgs_rgb))
        print(labels_rgb)
        print(labels_ir)
        break
        # for (label_rgb, label_ir) in zip(labels_rgb, labels_ir):
        #     print(label_rgb, label_ir)

