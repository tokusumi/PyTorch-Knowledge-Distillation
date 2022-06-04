import warnings
from copy import deepcopy

from mmcv.utils import Registry
from mmcv.runner import load_checkpoint
from mmcls.models import build_classifier
from mmcv import Config


KDS = Registry("kds")


def load_ckpt(model, ckpt_file, map_location="cpu"):
    checkpoint = load_checkpoint(model, ckpt_file, map_location=map_location)

    if "CLASSES" in checkpoint.get("meta", {}):
        CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        from mmcls.datasets import ImageNet

        warnings.simplefilter("once")
        warnings.warn(
            "Class names are not saved in the checkpoint's "
            "meta data, use imagenet by default."
        )
        CLASSES = ImageNet.CLASSES
    model.CLASSES = CLASSES


def build_teacher_classifier(cfg):
    """build classifier and implements additional process for teacher model"""
    model = build_classifier(Config.fromfile(cfg.config).model)
    load_ckpt(model, cfg.ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_kd_classifier(cfg, student):
    """Build Classifier for KD, which manages student and teacher models."""
    kd_cfg = deepcopy(cfg)
    kd_cfg["student"] = student
    model = KDS.build(kd_cfg)
    return model
