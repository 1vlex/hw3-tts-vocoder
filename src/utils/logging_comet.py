try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


def _flatten_dict(d, parent_key="", sep="/"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def init_comet(enabled, project, run_name, config):
    if not enabled or Experiment is None:
        return None

    exp = Experiment(project_name=project)
    exp.set_name(run_name)

    if isinstance(config, dict):
        exp.log_parameters(_flatten_dict(config))

    return exp


def log_metrics(run, metrics, step=None):
    if run is not None:
        if step is None:
            run.log_metrics(metrics)
        else:
            run.log_metrics(metrics, step=step)


def log_audio(run, key, audio_np, sample_rate, step=None, caption=""):
    if run is None:
        return

    file_name = f"{key.replace('/', '_')}_{0 if step is None else step}.wav"
    metadata = {"caption": caption} if caption else None

    run.log_audio(
        audio_data=audio_np,
        sample_rate=int(sample_rate),
        file_name=file_name,
        step=step,
        metadata=metadata,
    )