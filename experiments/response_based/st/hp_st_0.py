from kds.core.optuna_hook import Category

distil = dict(
    loss=dict(type="SoftTarget", T=Category(0.25, 0.5, 0.75, 1)),
)

optuna_config = dict(
    optimized="accuracy_top-1",
    direction="maximize",
    pruner=dict(type="HyperbandPruner"),
    n_trials=10,
)
runner = dict(max_epochs=5)
data = dict(
    workers_per_gpu=1,
)
