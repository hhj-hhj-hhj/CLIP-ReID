import os

import torch
from torch.cuda import amp
from config import cfg
import argparse
from datasets.make_dataloader_vi import make_dataloader
from model.make_model_vi import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger

# train_loader_rgb, train_loader_ir, train_loader_normal_rgb, train_loader_normal_ir, val_loader, num_query, num_classes_rgb, num_classes_ir, camera_num_rgb, camera_num_ir, view_num_rgb, view_num_ir = make_dataloader(cfg)

# print('camera_rgb number is : {}'.format(camera_num_rgb))
# print('camera_ir number is : {}'.format(camera_num_ir))
# print('view_rgb number is : {}'.format(view_num_rgb))
# print('view_ir number is : {}'.format(view_num_ir))
# print('num_rgb number is : {}'.format(num_classes_rgb))
# print('num_ir number is : {}'.format(num_classes_ir))
if __name__ == "__main__":

    # device = 'cuda'
    image_features = []
    labels = []
    cams = []

    train_loader_stage2_all, train_loader_stage2_rgb, train_loader_stage2_ir, train_loader_stage1_all, train_loader_stage1_rgb, train_loader_stage1_ir, val_loader, num_query, num_classes_all, num_classes_rgb, num_classes_ir, camera_num_all, camera_num_rgb, camera_num_ir, view_num_all, view_num_rgb, view_num_ir = make_dataloader(
        cfg)

    # model = make_model(cfg, num_class=num_classes_all, camera_num=camera_num_all, view_num=view_num_all)

    max_len = len(train_loader_stage1_all)
    # 构造一个长度为10的随机列表，里面的值为0-max_len-1
    random_list = torch.randperm(max_len)
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1_all):
            img = img
            target = vid
            cam = target_cam
            with amp.autocast(enabled=True):
                # image_feature = model(img, target, get_image=True)
                for i, img_feat, cam_id in zip(target, img, cam):
                    labels.append(i)
                    # image_features.append(img_feat.cpu())
                    cams.append(cam_id.cpu())
        labels_list_all = torch.stack(labels, dim=0)  # N
        # image_features_list_all = torch.stack(image_features, dim=0)
        cam_list_all = torch.stack(cams, dim=0)

        print('labels_list_all shape is : {}'.format(labels_list_all.shape))
        # print('image_features_list_all shape is : {}'.format(image_features_list_all.shape))
        print('cam_list_all shape is : {}'.format(cam_list_all.shape))

        cam2modal = {
            0: 1,
            1: 1,
            2: 1,
            3: 0,
            4: 1,
            5: 1,
            6: 0
        }
    num_image_all = labels_list_all.shape[0]
    iter_list_all = torch.randperm(num_image_all)
    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    i_ter_all = num_image_all // batch

    for i in range(i_ter_all + 1):
        if i != i_ter_all:
            b_list_all = iter_list_all[i * batch:(i + 1) * batch]
        else:
            b_list_all = iter_list_all[i * batch:num_image_all]

        target = labels_list_all[b_list_all]
        # image_features = image_features_list_all[b_list_all]
        cam = cam_list_all[b_list_all]

        # for i,cam_id in enumerate(cam):
        #     if i == 10:
        #         break
        #     print(cam_id)
    cam_modal = torch.tensor([cam2modal[int(i)] for i in cam])

    print('cam_modal is : {}'.format(cam_modal.shape))
    print('target is : {}'.format(target.shape))
    # print('image_features is : {}'.format(image_features.shape))

    for i in range(len(cam_modal)):
        print(cam_modal[i], target[i], end = '\n------')



# import torch
# a = [1,2,3,4]
# b = torch.stack(a, dim=0)
# print(a)
# print(b)
