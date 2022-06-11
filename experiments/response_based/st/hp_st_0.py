from kds.core.optuna_hook import Category

distil = dict(
    loss=dict(type="SoftTarget", T=Category(0.25, 0.5, 0.75, 1)),
)
