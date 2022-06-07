# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Optional

import mmcv
from mmcv import Config

from mmcls.datasets import build_dataset
import typer

from arguments import (
    Metrics,
    t_config,
    t_out,
    t_metrics,
)


def main(
    config: Path = t_config,
    results: Path = typer.Argument(
        ..., help="the output file of test.py with --out option"
    ),
    out: Optional[Path] = t_out,
    metrics: List[Metrics] = t_metrics,
):
    """Evaluate metric of the " "results saved in pkl format"""

    outputs = mmcv.load(results)
    assert (
        "class_scores" in outputs
    ), 'No "class_scores" in result file, please set "--out-items" in test.py'

    cfg = Config.fromfile(config)
    assert metrics, 'Please specify at least one metric the argument "--metrics".'

    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    pred_score = outputs["class_scores"]

    eval_kwargs = cfg.get("evaluation", {}).copy()
    # hard-code way to remove EvalHook args
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=[m.value for m in metrics], metric_options={}))
    eval_res = dataset.evaluate(pred_score, **eval_kwargs)
    print(eval_res)

    if out:
        print(f"\ndumping results to {out}")
        mmcv.dump(eval_res, out)


if __name__ == "__main__":
    typer.run(main)
