import os

MMCONFIG = os.getenv("MMCONFIG")
_base_ = [
    f"{MMCONFIG}/_base_/models/resnet18_cifar.py",
    f"{MMCONFIG}/_base_/datasets/cifar10_bs16.py",
    f"{MMCONFIG}/_base_/schedules/cifar10_bs128.py",
    f"{MMCONFIG}/_base_/default_runtime.py",
]

data = dict(
    samples_per_gpu=128,  # For one GPU
    workers_per_gpu=2,
)

log_config = dict(interval=100)

distil = dict(
    type="ResponseBased",
    teacher=dict(
        config=f"{MMCONFIG}/resnet/resnet50_8xb16_cifar10.py",
        ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth",
    ),
    loss=dict(type="Logits", lambda_kd=1.0),
)
