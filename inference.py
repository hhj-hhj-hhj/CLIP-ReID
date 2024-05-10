import os
import time
import torch
from config import cfg
import argparse
from datasets.make_dataloader_vi import make_dataloader
from model.make_model_vi import make_model
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
# from torch.autograd import Variable
from inference_require.cclnet_dataloader import TestData
from inference_require.eval_metrics import eval_sysu
from inference_require.data_manager import process_query_sysu, process_gallery_sysu


feat_dim = 512

def extract_gall_feat(gall_loader):
    model.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.to('cuda')

            feat = model(x = input, get_image = True)

            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    print("Extracting Time:\t {:.3f}".format(time.time() - start))
    return gall_feat

def extract_query_feat(query_loader):
    model.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.to('cuda')
            feat = model(x = input, get_image = True)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--img_w', default=128, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=256, type=int,
                        metavar='imgh', help='img height')
    parser.add_argument('--test-batch-size', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader_stage2_all, train_loader_stage2_rgb, train_loader_stage2_ir, train_loader_stage1_all, train_loader_stage1_rgb, train_loader_stage1_ir, val_loader, num_query, num_classes_all, num_classes_rgb, num_classes_ir, camera_num_all, camera_num_rgb, camera_num_ir, view_num_all, view_num_rgb, view_num_ir = make_dataloader(
            cfg)

    print('==> Resuming from checkpoint..')

    print('----load checkpoint-----')

    model = make_model(cfg, num_class=num_classes_rgb, camera_num=camera_num_rgb + camera_num_ir, view_num = view_num_rgb)
    model.load_param(cfg.TEST.WEIGHT)

    data_path = "E:\\hhj\\SYSU-MM01"

    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)
    nquery = len(query_label)
    ngall = len(gall_label)

    # print("Dataset {} Statistics:".format(args.dataset))
    # print("  ----------------------------")
    # print("  subset   | # ids | # images")
    # print("  ----------------------------")
    # print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    # print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    # print("  ----------------------------")


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    transform_visible = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    print("Data loading time:\t {:.3f}".format(time.time() - end))
    print('----------Testing----------')

    query_feat = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_visible, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

        gall_feat = extract_gall_feat(trial_gall_loader)
        distmat = np.matmul(query_feat, np.transpose(gall_feat))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

        print('Test Trial: {}'.format(trial))
        print("Performance: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print("-----------------------Next Trial--------------------")


