import os

MMCONFIG = os.getenv("MMCONFIG")
_base_ = [
    f"{MMCONFIG}/_base_/models/resnet18_cifar.py",
    f"{MMCONFIG}/_base_/datasets/cifar100_bs16.py",
    f"{MMCONFIG}/_base_/schedules/cifar10_bs128.py",
    f"{MMCONFIG}/_base_/default_runtime.py",
]

model = dict(head=dict(num_classes=100))

data = dict(
    samples_per_gpu=128,  # For one GPU
    workers_per_gpu=2,
)

optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy="step", step=[60, 120, 160], gamma=0.2)
log_config = dict(interval=100)

distil = dict(
    type="ResponseBased",
    teacher=dict(
        config=f"{MMCONFIG}/resnet/resnet50_8xb16_cifar100.py",
        ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth",
    ),
    loss=dict(type="Logits", lambda_kd=1.0),
)
