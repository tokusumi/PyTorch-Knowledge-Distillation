import torch.nn as nn
import torch.nn.functional as F
from mmcls.models import LOSSES


@LOSSES.register_module()
class SoftTarget(nn.Module):
    """
    type: ResponseBased

    loss: CrossEntropy(P, Q) * T^2, where
        T: temparature
        Q, P: softmax with temparature T of teacher/student logits

    Ref:
        - Distilling the Knowledge in a Neural Network (2015)
        - Geoffrey Hinton, Oriol Vinyals, Jeff Dean
        - https://arxiv.org/pdf/1503.02531.pdf

    """

    def __init__(self, T=1.0):
        """
        Args:
            T (float): = [1, inf). Temparature.

        Temparature:
            - high: much negative logits is ignored. In the higher limit, equivalent to Logits loss
            - low: pay attention all logits, including very noisy logits else.
        """
        super(SoftTarget, self).__init__()
        self.T = T
        self.T2 = T * T

    def forward(self, out_s, out_t):
        """out: logits (before softmax)"""

        # F.cross_entropy doesn't suit on here because
        # it doesn't support to calculate softmax with temparature
        # and also, pass the value of pre-calculated softmax-form prediction.

        loss_pointwise = (
            -1 * F.softmax(out_t / self.T, dim=1) * F.log_softmax(out_s / self.T, dim=1)
        )
        loss = loss_pointwise.sum() / out_s.size(0)

        # rescale by T**2 to fit scale of hard label
        return loss * self.T2
