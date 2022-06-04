from pathlib import Path

from mmcls.models import ImageClassifier
from mmcls.utils import get_root_logger
from mmcv.runner import build_optimizer, build_runner, load_checkpoint

from kds.baseline.kd_module import SimpleClassifierKD


def test_base_separated():
    base_kd = SimpleClassifierKD(
        student=ImageClassifier(backbone=dict(type="EfficientNet", arch="b0")),
        teacher=ImageClassifier(backbone=dict(type="EfficientNet", arch="b1")),
    )

    # teacher model is eval mode
    assert not base_kd.teacher.training

    # prediction for eval only rely on student model


def test_type_one():
    base_kd = SimpleClassifierKD(
        student=ImageClassifier(backbone=dict(type="EfficientNet", arch="b0")),
        teacher=ImageClassifier(backbone=dict(type="EfficientNet", arch="b1")),
    )

    # prediction return only student output

    # training can work


def test_ckpt_save_load_only_student(tmp_path: Path):
    work_dir = tmp_path
    student = ImageClassifier(backbone=dict(type="EfficientNet", arch="b0"))
    model = SimpleClassifierKD(
        student=student,
        teacher=ImageClassifier(backbone=dict(type="EfficientNet", arch="b1")),
    )

    optimizer = build_optimizer(
        model, dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
    )

    cfg_runner = {"type": "EpochBasedRunner", "max_epochs": 2}

    runner = build_runner(
        cfg_runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=str(work_dir),
            logger=get_root_logger(),
            # meta=meta)
        ),
    )
    # can save something
    runner.save_checkpoint(out_dir=str(work_dir), filename_tmpl="epoch_{}.pth")
    ckpt = work_dir / "epoch_1.pth"
    assert ckpt.is_file()

    # load ckpt into only student model without ClassifierKD instanse
    pred_student = ImageClassifier(backbone=dict(type="EfficientNet", arch="b0"))
    import pdb

    pdb.set_trace()
    # pred_student != student
    checkpoint = load_checkpoint(
        pred_student,
        str(ckpt),
        map_location="cpu",
        revise_keys=[(r"^module\.", ""), (r"^student\.", "")],
    )
    # pred_student == student
