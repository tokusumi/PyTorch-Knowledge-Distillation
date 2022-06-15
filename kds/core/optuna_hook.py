from mmcv.runner import HOOKS, Hook, BaseRunner
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.evaluation import DistEvalHook, EvalHook

import optuna
from optuna.trial import Trial


@HOOKS.register_module()
class OptunaTrialHook(Hook):
    def __init__(self, trial, optimized="accuracy_top-1", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.optimized = optimized

    @master_only
    def before_run(self, runner: BaseRunner):
        super(OptunaTrialHook, self).before_run(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

    def after_train_epoch(self, runner: BaseRunner):
        epoch = runner.epoch
        acc = None
        if self.optimized == "loss":
            acc = runner.outputs.get("loss", 0)
        else:
            eval_res = self.eval_hook.dataloader.dataset.evaluate(
                self.eval_hook.latest_results, **self.eval_hook.eval_kwargs
            )
            for key, value in eval_res.items():
                if key == self.optimized:
                    acc = value
                    break
        if acc is None:
            return

        self.trial.report(acc, epoch)
        self.trial.set_user_attr("last", float(acc))
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def after_iter(self, runner):
        pass


class BaseSuggest:
    def override(self, trial, key):
        pass


class Category(BaseSuggest):
    def __init__(self, *args):
        self.args = args

    def override(self, trial: Trial, key: str):
        return trial.suggest_categorical(key, choices=self.args)


class DiscreteUniform(BaseSuggest):
    def __init__(self, low, high, q):
        self.low = low
        self.high = high
        self.q = q

    def override(self, trial: Trial, key: str):
        return trial.suggest_discrete_uniform(
            key, low=self.low, high=self.high, q=self.q
        )


class Float(BaseSuggest):
    def __init__(self, low, high, step=None, log=None):
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def override(self, trial: Trial, key: str):
        return trial.suggest_float(
            key, low=self.low, high=self.high, step=self.step, log=self.log
        )


class Int(BaseSuggest):
    def __init__(self, low, high, step=None, log=None):
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def override(self, trial: Trial, key: str):
        return trial.suggest_int(
            key, low=self.low, high=self.high, step=self.step, log=self.log
        )


class LogUniform(BaseSuggest):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def override(self, trial: Trial, key: str):
        return trial.suggest_loguniform(key, low=self.low, high=self.high)


class Uniform(BaseSuggest):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def override(self, trial: Trial, key: str):
        return trial.suggest_uniform(key, low=self.low, high=self.high)
