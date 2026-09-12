"""Optional experiment tracking.

Training runs can stream metrics to Weights & Biases or SwanLab, but neither is
required to train or evaluate. Tracking is off unless ``--track`` asks for it,
so a fresh clone runs without any account, API key or network access, and never
uploads to someone else's workspace.

``log`` is safe to call whether or not ``init`` ran, which keeps the call sites
in the engines free of backend checks.
"""
import os

_active = []


def init(backend, project, experiment_name, config, workspace=None):
    """Start a tracking run. ``backend`` is one of none / wandb / swanlab / both."""
    if backend in (None, "", "none"):
        return

    if backend in ("wandb", "both"):
        import wandb
        wandb.init(project=project, name=experiment_name, config=config)
        _active.append("wandb")

    if backend in ("swanlab", "both"):
        import swanlab
        swanlab.init(
            project=project,
            workspace=workspace or os.getenv("SWANLAB_WORKSPACE"),
            experiment_name=experiment_name,
            config=config,
        )
        _active.append("swanlab")


def log(metrics, step=None):
    """Forward a metrics dict to whichever backends are active."""
    if not _active:
        return
    if "wandb" in _active:
        import wandb
        wandb.log(metrics, step=step)
    if "swanlab" in _active:
        import swanlab
        swanlab.log(metrics, step=step)


def finish():
    """Close out the active runs."""
    if "wandb" in _active:
        import wandb
        wandb.finish()
    if "swanlab" in _active:
        import swanlab
        swanlab.finish()
    del _active[:]
