import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.sysu_mm01 import SYSUMM01
from datasets.bases import ImageDataset

from config import cfg

from timm.data.random_erasing import RandomErasing
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF

dataset = SYSUMM01(root=r'D:\Re-ID_Dataset\person\Infrared')

train_transforms = T.Compose([
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

train_normol_transforms = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
    T.ToTensor()
])


train_set_rgb = ImageDataset(dataset.train_rgb, train_normol_transforms)
train_set_ir = ImageDataset(dataset.train_ir, train_normol_transforms)

num_classes_rgb = dataset.num_train_rgb_pids
num_classes_ir = dataset.num_train_ir_pids
cam_num_rgb = dataset.num_train_rgb_cams
cam_num_ir = dataset.num_train_ir_cams
view_num_rgb = dataset.num_train_rgb_vids
view_num_ir = dataset.num_train_ir_vids


print(num_classes_ir,num_classes_rgb)


# writer = SummaryWriter('rgb')

for i,data in enumerate(train_set_rgb):
    if i == 5:
        break
    print('-----------------')
    img, pid, camid, trackid, img_path = data
    TF.to_pil_image(img).show()
    print(f'rgb模态{pid}号行人{camid}号相机{trackid}号视角，路径{img_path}')
    # writer.add_image('rgb', img, i)

for i,data in enumerate(train_set_ir):
    if i == 5:
        break
    print('-----------------')
    img, pid, camid, trackid, img_path = data
    TF.to_pil_image(img).show()
    print(f'ir模态{pid}号行人{camid}号相机{trackid}号视角，路径{img_path}')
    # writer.add_image('ir', img, i + 10)

# writer.add_image('rgb', train_set_rgb[-1][0], 10)
# writer.close()