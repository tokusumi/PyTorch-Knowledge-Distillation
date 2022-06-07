# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path
from typing import List, Optional
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import get_root_logger, setup_multi_processes
import typer

from arguments import (
    OutItems,
    Metrics,
    Launcher,
    Device,
    t_config,
    t_checkpoint,
    t_out,
    t_out_items,
    t_metrics,
    t_show,
    t_show_dir,
    t_gpu_collect,
    t_tmpdir,
    t_gpu_id,
    t_launcher,
    t_local_rank,
    t_device,
)


def main(
    config: Path = t_config,
    checkpoint: Path = t_checkpoint,
    out: Optional[Path] = t_out,
    out_items: List[OutItems] = t_out_items,
    metrics: List[Metrics] = t_metrics,
    show: bool = t_show,
    show_dir: Optional[Path] = t_show_dir,
    gpu_collect: bool = t_gpu_collect,
    tmpdir: Optional[Path] = t_tmpdir,
    gpu_id: int = t_gpu_id,
    launcher: Launcher = t_launcher,
    local_rank: str = t_local_rank,
    device: Device = t_device,
):
    """mmcls test model"""
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = local_rank

    assert (
        metrics or out
    ), "Please specify at least one of output path and evaluation metrics."

    cfg = mmcv.Config.fromfile(config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    cfg.gpu_ids = [gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if launcher == Launcher.none:
        distributed = False
    else:
        distributed = True
        init_dist(launcher.value, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if device == Device.ipu else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update(
        {
            k: v
            for k, v in cfg.data.items()
            if k
            not in [
                "train",
                "val",
                "test",
                "train_dataloader",
                "val_dataloader",
                "test_dataloader",
            ]
        }
    )
    test_loader_cfg = {
        **loader_cfg,
        "shuffle": False,  # Not shuffle by default
        "sampler_cfg": None,  # Not use sampler by default
        **cfg.data.get("test_dataloader", {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        str(checkpoint),
        map_location="cpu",
        revise_keys=[(r"^module\.", ""), (r"^student\.", "")],
    )

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

    if not distributed:
        if device == Device.cpu:
            model = model.cpu()
        elif device == Device.ipu:
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper

            opts = cfg2options(cfg.runner.get("options_cfg", {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
            data_loader.init(opts["inference"])
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if not model.device_ids:
                assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), (
                    "To test with CPU, please confirm your mmcv version "
                    "is not lower than v1.4.4"
                )
        model.CLASSES = CLASSES
        # show_kwargs = {} if show_options is None else show_options
        show_kwargs = {}
        outputs = single_gpu_test(model, data_loader, show, show_dir, **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, tmpdir, gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        logger = get_root_logger()
        if metrics:
            eval_results = dataset.evaluate(
                results=outputs,
                metric=[m.value for m in metrics],
                # metric_options=metric_options,
                metric_options={},
                logger=logger,
            )
            results.update(eval_results)
            for k, v in eval_results.items():
                if isinstance(v, np.ndarray):
                    v = [round(out, 2) for out in v.tolist()]
                elif isinstance(v, Number):
                    v = round(v, 2)
                else:
                    raise ValueError(f"Unsupport metric type: {type(v)}")
                print(f"\n{k} : {v}")
        if out:
            if OutItems.none not in out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    "class_scores": scores,
                    "pred_score": pred_score,
                    "pred_label": pred_label,
                    "pred_class": pred_class,
                }
                if OutItems.all in out_items:
                    results.update(res_items)
                else:
                    for key in out_items:
                        results[key.value] = res_items[key.value]
            print(f"\ndumping results to {out}")
            mmcv.dump(results, out)


if __name__ == "__main__":
    typer.run(main)
