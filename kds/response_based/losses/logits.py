import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import LOSSES


@LOSSES.register_module()
class Logits(nn.Module):
    """
    type: ResponseBased

    Ref:
        Do Deep Nets Really Need to be Deep?
        http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    """

    def __init__(self, lambda_kd=1.0):
        super(Logits, self).__init__()
        self.lambda_kd = lambda_kd

    def forward(self, out_s, out_t):
        """out: logits (before softmax)"""
        loss = F.mse_loss(out_s, out_t) * self.lambda_kd

        return loss
