# Copyright (c) OpenMMLab. All rights reserved.
"""
This is fully copied from `mmclassification/tools/train.py` and added only 3 lines:

```
from kds import build_kd_classifier

...

def main():
    ...
    (Line. 191)
    if hasattr(cfg, "distil"):
        # wrap student model and teacher model to train in the KD way
        model = build_kd_classifier(cfg.distil, model)
    ...
```
"""
import copy
import os
import os.path as osp
from pathlib import Path
import time
from typing import Optional

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, ConfigDict
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger, setup_multi_processes
import typer

from kds import build_kd_classifier
from kds.core.optuna_hook import BaseSuggest
from arguments import (
    Launcher,
    Device,
    t_config,
    t_gpu_id,
    t_launcher,
    t_local_rank,
    t_device,
)


def update_config_with_optuna(trial, cfg: Config, optuna_config: Config):
    # override ConfDict with user-defined optuna config as:
    # cfg.distil.alpha = trial.suggest_categorical("distil.alpha", [0., 0.25, 0.5, 0.75, 1.])
    # from (cfg.optuna_config)
    # distil = dict(
    #               alpha=("suggest_categorical", [0., 0.25, 0.5, 0.75, 1.]),
    # )
    optuna_config = copy.deepcopy(optuna_config)

    def dig(nested_config, prefix=""):
        for key, value in nested_config.items():
            if isinstance(value, dict):
                dig(value, prefix=".".join([prefix, key]))
            elif isinstance(value, BaseSuggest):
                nested_config[key] = value.override(trial, key=".".join([prefix, key]))

    # convert optuna_config value
    dig(optuna_config)
    cfg.merge_from_dict(optuna_config)
    return cfg


def main(
    config: Path = t_config,
    optuna_config: Path = typer.Argument(
        ..., help="experiments optuna config file path"
    ),
    work_dir: Optional[Path] = typer.Argument(
        None, help="the dir to save logs and models"
    ),
    device: Device = t_device,
    gpu_id: int = t_gpu_id,
    ipu_replicas: Optional[int] = typer.Option(None, help="num of ipu replicas to use"),
    seed: int = typer.Option(None, help="random seed"),
    diff_seed: bool = typer.Option(
        False,
        help="Whether or not set different seeds for different ranks",
    ),
    deterministic: bool = typer.Option(
        False,
        help="whether to set deterministic options for CUDNN backend.",
    ),
    launcher: Launcher = t_launcher,
    local_rank: str = t_local_rank,
):
    """Train a model"""
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = local_rank

    cfg = Config.fromfile(config)
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if work_dir is not None:
        # update configs according to CLI args if work_dir is not None
        cfg.work_dir = str(work_dir)
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(config.absolute().parent.name)[0]
        )
    cfg.gpu_ids = [gpu_id]

    if ipu_replicas is not None:
        cfg.ipu_replicas = ipu_replicas
        device = Device.ipu

    # init distributed env first, since logger depends on the dist info.
    if launcher == Launcher.none:
        distributed = False
    else:
        distributed = True
        init_dist(launcher.value, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    optuna_config = Config.fromfile(optuna_config)

    def objective(_cfg, _optuna_config, _seed):
        def _objective(trial):
            cfg = copy.deepcopy(_cfg)
            cfg = update_config_with_optuna(trial, cfg, _optuna_config)

            cfg.work_dir = osp.join(cfg.work_dir, trial.trial_id)

            # create work_dir
            mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
            # dump config
            cfg.dump(osp.join(cfg.work_dir, config.absolute().name))
            # init the logger before other steps
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
            logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

            # init the meta dict to record some important information such as
            # environment info and seed, which will be logged
            meta = dict()
            # log env info
            env_info_dict = collect_env()
            env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
            dash_line = "-" * 60 + "\n"
            logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
            meta["env_info"] = env_info

            # log some basic info
            logger.info(f"Distributed training: {distributed}")
            logger.info(f"Config:\n{cfg.pretty_text}")

            # set random seeds
            seed = init_random_seed(_seed)
            seed = seed + dist.get_rank() if diff_seed else seed
            logger.info(
                f"Set random seed to {seed}, " f"deterministic: {deterministic}"
            )
            set_random_seed(seed, deterministic=deterministic)
            cfg.seed = seed
            meta["seed"] = seed

            model = build_classifier(cfg.model)
            model.init_weights()

            if hasattr(cfg, "distil"):
                # Init teacher
                # wrap student model and teacher model to train in the KD way
                model = build_kd_classifier(cfg.distil, model)

            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))

            # save mmcls version, config file content and class names in
            # runner as meta data
            meta.update(
                dict(
                    mmcls_version=__version__,
                    config=cfg.pretty_text,
                    CLASSES=datasets[0].CLASSES,
                )
            )

            # Inject trial-pruner in training loop
            if not cfg.get("custom_hooks"):
                cfg["custom_hooks"] = []
            cfg["custom_hooks"].append(
                dict(type="OptunaTrialHook", trial=trial, priority="VERY_LOW")
            )

            # add an attribute for visualization convenience
            train_model(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=True,
                timestamp=timestamp,
                device=device.value,
                meta=meta,
            )
            # must get acc from runner
            return 0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective(cfg, optuna_config, seed), n_trials=100)
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    typer.run(main)