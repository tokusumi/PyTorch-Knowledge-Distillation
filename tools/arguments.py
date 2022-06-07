from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer


class OutItems(Enum):
    class_scores = "class_scores"
    pred_score = "pred_score"
    pred_label = "pred_label"
    pred_class = "pred_class"
    none = "none"
    all = "all"


class Metrics(Enum):
    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    f1_score = "f1_score"


class Launcher(Enum):
    none = "none"
    pytorch = "pytorch"
    slurm = "slurm"
    mpi = "mpi"


class Device(Enum):
    cpu = "cpu"
    cuda = "cuda"
    ipu = "ipu"


t_config: Path = typer.Argument(..., help="experiments config file path")
t_checkpoint: Path = typer.Argument(..., help="checkpoint file")
t_out: Optional[Path] = typer.Option(None, help="output result file")
t_out_items: List[OutItems] = typer.Option(
    ["all"],
    help="Besides metrics, what items will be included in the output "
    "result file. You can choose some of metrics, "
    'or use "all" to include all above, or use "none" to disable all of '
    "above. Defaults to output all.",
)
t_metrics: List[Metrics] = typer.Option(
    ["accuracy"], help="evaluation metrics, which depends on the dataset"
)
t_show: bool = typer.Option(False, help="show results")
t_show_dir: Optional[Path] = typer.Option(
    None, help="directory where painted images will be saved"
)
t_gpu_collect: bool = typer.Option(False, help="whether to use gpu to collect results")
t_tmpdir: Optional[Path] = typer.Option(None, help="tmp dir for writing some results")
t_gpu_id: int = typer.Option(
    0, help="id of gpu to use " "(only applicable to non-distributed testing)"
)
t_launcher: Launcher = typer.Option("none", help="job launcher")
t_local_rank: str = typer.Option("0")
t_device: Device = typer.Option("cuda", help="device used for testing")
