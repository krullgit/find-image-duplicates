# --------------------------------------------------------
# Duplicate Detection
# Licensed under The GPL2 License [see LICENSE for details]
# Written by Matthes Krull
# --------------------------------------------------------

import yaml
from yacs.config import CfgNode as CN
import os


_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Delete every image which has fewer pixels than this.
_C.DATA.IMG_MIN_AREA = 2500

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Resize width and hight before processing. Also effects system memory used.
_C.AUG.IMG_RESCALE_SIZE = 200
# A list of blurs which is applied to the images
_C.AUG.GAUSSIAN_BLUR_RADIUS_LIST = [5]
# Amount of pixels for each side, which get blacked [left, top, right, bottom]
_C.AUG.BLACK_MASK = [0, 0, 0, 0]

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Only contours that are bigger that this (percentage) are considered for the image difference
_C.MODEL.IMG_CONTOUR_THRESHOLD = 0.01  # range: [0, 1]
# Every image that has a difference (percentage) lower than this will be deleted
_C.MODEL.IMG_CHANGE_THRESHOLD = 0.02  # range: [0, 1]


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):

    config.defrost()

    if args.cfg:
        _update_config_from_file(config, args.cfg)
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.path_dataset:
        config.DATA.PATH_DATASET = args.path_dataset

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
