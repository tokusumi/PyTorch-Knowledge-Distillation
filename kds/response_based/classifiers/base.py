from typing import Union

from mmcls.models import BaseClassifier, ImageClassifier
from mmcv import Config, ConfigDict
from mmcls.models import build_classifier
from mmcls.models import LOSSES
import torch.nn as nn

from kds.core.builder import KDS, build_teacher_classifier


class BaseClassifierKD(BaseClassifier):
    """Handler for teacher and student classifier"""

    def __init__(
        self,
        student: Union[ImageClassifier, ConfigDict],
        teacher: Union[ImageClassifier, ConfigDict],
        loss: Union[nn.Module, ConfigDict],
    ):
        """
        Args:
            student: same as "model" in config
            teacher: Config format are {config: <model config file path>, ckpt: <checkpoint for teacher model>}
            loss: Config format are {type: <select KD loss>, ...} as same as model loss config
        """
        super(BaseClassifier, self).__init__(None)
        # build classifier
        if isinstance(student, ConfigDict):
            student = build_classifier(student)
            student.init_weights()
        self.student = student

        # build classifier for kd
        if isinstance(teacher, ConfigDict):
            teacher = build_teacher_classifier(teacher)

        self.teacher = teacher

        # build kd loss
        if isinstance(loss, ConfigDict):
            loss = LOSSES.build(loss)
        self.loss = loss

        self.augments = None
        if self.student.augments is not None:
            # get augment function to apply it with teacher model
            self.augments = self.student.augments
            self.student.augments = None

    def simple_test(self, img, **kwargs):
        return self.student.simple_test(img, **kwargs)

    def extract_feat(self, img, stage="neck"):
        return self.student.extract_feat(img, stage=stage)


@KDS.register_module()
class ResponseBased(BaseClassifierKD):
    """Handler for teacher and student classifier"""

    def forward_train(self, img, gt_label, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        # calculate class loss
        pred_student = self.student.extract_feat(img)
        pred_student = self.student.head.simple_test(
            pred_student, softmax=False, post_process=False
        )
        losses = self.student.head.loss(pred_student, gt_label)
        cls_loss = losses.pop("loss")
        losses["cls_loss"] = cls_loss

        # calculate kd losses
        pred_teacher = self.teacher.simple_test(img, softmax=False, post_process=False)
        kd_loss = self.loss(pred_student, pred_teacher)
        losses["kd_loss"] = kd_loss

        # will be added both loss in _parse_losses
        # losses["loss"] = kd_loss + cls_loss
        return losses
