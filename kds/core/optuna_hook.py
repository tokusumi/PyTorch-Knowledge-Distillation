from mmcv.runner import HOOKS, Hook

import optuna
from optuna.trial import Trial


@HOOKS.register_module()
class OptunaTrialHook(Hook):
    def __init__(self, trial):
        self.trial = trial

    def after_epoch(self, runner):
        import pdb

        pdb.set_trace()
        epoch = runner.epoch
        acc = 0

        self.trial.report(acc, epoch)
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
