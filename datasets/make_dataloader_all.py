import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .sysu_mm01 import SYSUMM01, SYSUData_Stage2
from utils.utils import GenIdx, IdentitySampler_nosk

__factory = {
    'sysu_mm01': SYSUMM01
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader_all(cfg):
    transforms_train = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    transform_train_rgb = T.Compose([
        T.ToPILImage(),
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    transform_train_ir = T.Compose([
        T.ToPILImage(),
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    # transform_train_rgb = T.Compose([
    #     T.ToPILImage(),
    #     T.RandomGrayscale(p=0.5),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop(cfg.INPUT.SIZE_TEST),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     T.RandomErasing(p=0.5)
    # ])
    # transform_train_ir = T.Compose([
    #     T.ToPILImage(),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop(cfg.INPUT.SIZE_TEST),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     T.RandomErasing(p=0.5),
    # ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_sp_set = SYSUData_Stage2(cfg.DATASETS.ROOT_DIR, transform_train_rgb, transform_train_ir)
    color_pos, thermal_pos = GenIdx(train_sp_set.train_color_label, train_sp_set.train_thermal_label)

    sampler = IdentitySampler_nosk(train_sp_set.train_color_label, train_sp_set.train_thermal_label, color_pos,
                                   thermal_pos,
                                   cfg.SOLVER.STAGE3.NUM_INSTANCES, cfg.SOLVER.STAGE3.BATCH_SIZE)

    train_sp_set.cIndex = sampler.index1
    train_sp_set.tIndex = sampler.index2

    train_sp_loader = DataLoader(train_sp_set,
                                 batch_size=cfg.SOLVER.STAGE3.BATCH_SIZE * cfg.SOLVER.STAGE3.NUM_INSTANCES,
                                 sampler=sampler,
                                 num_workers=0,
                                 drop_last=True)
    return train_sp_loader