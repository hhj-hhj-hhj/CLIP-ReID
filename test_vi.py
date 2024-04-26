import os
from config import cfg
import argparse
from datasets.make_dataloader_vi import make_dataloader
from model.make_model_vi import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger
from my_test import do_my_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader_stage2_all, train_loader_stage2_rgb, train_loader_stage2_ir, train_loader_stage1_all, train_loader_stage1_rgb, train_loader_stage1_ir, val_loader, num_query, num_classes_all, num_classes_rgb, num_classes_ir, camera_num_all, camera_num_rgb, camera_num_ir, view_num_all, view_num_rgb, view_num_ir = make_dataloader(
        cfg)
    # 这里需要传入相机数，还不确定传入的相机数是否要将红外和可将光的相机数相加
    model = make_model(cfg, num_class=num_classes_rgb, camera_num=camera_num_rgb + camera_num_ir, view_num = view_num_rgb)
    model.load_param(cfg.TEST.WEIGHT)

    do_inference(cfg,
                 model,
                 val_loader,
                 num_query)


