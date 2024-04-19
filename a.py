import os

import torch

from config import cfg
import argparse
from datasets.make_dataloader_vi import make_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger

# train_loader_rgb, train_loader_ir, train_loader_normal_rgb, train_loader_normal_ir, val_loader, num_query, num_classes_rgb, num_classes_ir, camera_num_rgb, camera_num_ir, view_num_rgb, view_num_ir = make_dataloader(cfg)
#
# print('camera_rgb number is : {}'.format(camera_num_rgb))
# print('camera_ir number is : {}'.format(camera_num_ir))
# print('view_rgb number is : {}'.format(view_num_rgb))
# print('view_ir number is : {}'.format(view_num_ir))
# print('num_rgb number is : {}'.format(num_classes_rgb))
# print('num_ir number is : {}'.format(num_classes_ir))
# if __name__ == "__main__":
#     train_loader_stage2_rgb, train_loader_stage2_ir, train_loader_stage1_rgb, train_loader_stage1_ir, val_loader, num_query, num_classes_rgb, num_classes_ir, camera_num_rgb, camera_num_ir, view_num_rgb, view_num_ir = make_dataloader(cfg)
#
#     with torch.no_grad():
#         for n_iter, (img, vid, target_cam, target_view, path) in enumerate(train_loader_stage1_rgb):
#             if n_iter == 10:
#                 break
#             print(img.shape,n_iter)

import torch
a = [1,2,3,4]
b = torch.stack(a, dim=0)
print(a)
print(b)
