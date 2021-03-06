import os

MMCONFIG = os.getenv("MMCONFIG")
_base_ = [
    f"{MMCONFIG}/_base_/models/resnet18.py",
    f"{MMCONFIG}/_base_/datasets/cifar10_bs16.py",
    f"{MMCONFIG}/_base_/default_runtime.py",
]

model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth",
            prefix="backbone",
        ),
    ),
    head=dict(num_classes=10),
)

img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False,
)
train_pipeline = [
    dict(type="RandomCrop", size=32, padding=4),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Resize", size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="Resize", size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=128,  # For one GPU
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# lr is set for a batch size of 128
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", step=[15])
runner = dict(type="EpochBasedRunner", max_epochs=200)
log_config = dict(interval=100)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(save_best="auto")

distil = dict(
    type="ResponseBased",
    teacher=dict(
        config="./experiments/baseline/resnet50.py",
        ckpt="./work_dirs/resnet50/latest.pth",
    ),
    loss=dict(type="Logits", lambda_kd=1.0),
)
