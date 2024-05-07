import glob
import re
import random
import os.path as osp
import os
from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class SYSUMM01(BaseImageDataset):

    dataset_dir = 'SYSU-MM01'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(SYSUMM01, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.pid_begin = pid_begin

        gallery = self._process_galler(self.dataset_dir)
        query = self._process_query(self.dataset_dir)
        train_rgb, train_ir = self._process_train(self.dataset_dir)
        train_all = train_rgb + train_ir




        if(verbose):
            print("=> SYSU-MM01 loaded")
            self.print_dataset_statistics(train_rgb, query, gallery)

        self.gallery = gallery
        self.query = query
        self.train_rgb = train_rgb
        self.train_ir = train_ir
        self.train_all = train_all

        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_train_rgb_pids, self.num_train_rgb_imgs, self.num_train_rgb_cams, self.num_train_rgb_vids = self.get_imagedata_info(self.train_rgb)
        self.num_train_ir_pids, self.num_train_ir_imgs, self.num_train_ir_cams, self.num_train_ir_vids = self.get_imagedata_info(self.train_ir)
        self.num_train_all_pids, self.num_train_all_imgs, self.num_train_all_cams, self.num_train_all_vids = self.get_imagedata_info(self.train_all)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))


    def _process_galler(self, dir_path, mode = 'all', trial = 0, relabel=False):

        random.seed(trial)

        if mode == 'all':
            rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        elif mode == 'indoor':
            rgb_cameras = ['cam1', 'cam2']

        file_path = os.path.join(dir_path, 'exp\\test_id.txt')
        files_rgb = []
        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        pid2label = {pid: label for label, pid in enumerate(ids)}
        dataset = []

        # for id in sorted(ids):
        #     for cam in rgb_cameras:
        #         img_dir = os.path.join(dir_path, cam, id)
        #         if os.path.isdir(img_dir):
        #             new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
        #             files_rgb.append(random.choice(new_files))

        for id in sorted(ids):
            for cam in rgb_cameras:
                img_dir = os.path.join(dir_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '\\' + i for i in os.listdir(img_dir)])
                    # cclnet这里是随机选择一个图片，但是这里我改成了全部图片
                    # files_rgb.append(random.choice(new_files))
                    files_rgb.extend(new_files)

        # gall_img = []
        # gall_id = []
        # gall_cam = []
        for img_path in files_rgb:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 0))
            # gall_img.append(img_path)
            # gall_id.append(pid)
            # gall_cam.append(camid)
        return dataset

    def _process_query(self, dir_path, mode = 'all', relabel=False):
        if mode == 'all':
            ir_cameras = ['cam3', 'cam6']
        elif mode == 'indoor':
            ir_cameras = ['cam3', 'cam6']

        file_path = os.path.join(dir_path, 'exp\\test_id.txt')
        files_rgb = []
        files_ir = []

        with open(file_path, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            ids = ["%04d" % x for x in ids]

        pid2label = {pid: label for label, pid in enumerate(ids)}
        dataset = []

        for id in sorted(ids):
            for cam in ir_cameras:
                img_dir = os.path.join(dir_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '\\' + i for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)
        # query_img = []
        # query_id = []
        # query_cam = []
        for img_path in files_ir:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _process_train(self, dir_path, relabel=True):

        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        ir_cameras = ['cam3', 'cam6']

        # load id info
        file_path_train = os.path.join(dir_path, 'exp\\train_id.txt')
        file_path_val = os.path.join(dir_path, 'exp\\val_id.txt')

        with open(file_path_train, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_train = ["%04d" % x for x in ids]

        with open(file_path_val, 'r') as file:
            ids = file.read().splitlines()
            ids = [int(y) for y in ids[0].split(',')]
            id_val = ["%04d" % x for x in ids]

        # combine train and val split
        # 这里不知道要不要合并train和val，先不注释掉
        id_train.extend(id_val)

        files_rgb = []
        files_ir = []
        for id in sorted(id_train):
            for cam in rgb_cameras:
                img_dir = os.path.join(dir_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '\\' + i for i in os.listdir(img_dir)])
                    files_rgb.extend(new_files)

            for cam in ir_cameras:
                img_dir = os.path.join(dir_path, cam, id)
                if os.path.isdir(img_dir):
                    new_files = sorted([img_dir + '\\' + i for i in os.listdir(img_dir)])
                    files_ir.extend(new_files)

        files_all = files_rgb.copy()
        files_all.extend(files_ir)

        # relabel
        pid_container = set()
        for img_path in files_ir:
            pid = int(img_path[-13:-9])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        fix_image_width = 144
        fix_image_height = 288

        dataset_rgb = []
        dataset_ir = []

        for img_path in files_rgb:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            dataset_rgb.append((img_path, pid, camid, 0))

        for img_path in files_ir:
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            dataset_ir.append((img_path, pid, camid, 0))

        return dataset_rgb, dataset_ir