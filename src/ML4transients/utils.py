import yaml
from datetime import datetime
from pathlib import Path

def realtime_update(config_path, status):
    """Update run_info with timestamps and status."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    now = datetime.utcnow().isoformat()
    run_info = config.setdefault("run_info", {})
    if "started" not in run_info:
        run_info["started"] = now
    run_info["finished"] = now
    run_info["status"] = status

    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def append_config(config):
    """Append full config snapshot (including run_info) to run_history."""
    output_dir = Path(config["path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    final_config_path = output_dir / "config_summary.yaml"

    if final_config_path.exists():
        with open(final_config_path) as f:
            frozen = yaml.safe_load(f)
    else:
        frozen = {}

    # Prepare run_history list
    if "run_history" not in frozen:
        frozen["run_history"] = []

    # Append full config snapshot minus previous run_history (avoid nesting)
    snapshot = {k: v for k, v in config.items() if k != "run_history"}
    frozen["run_history"].append(snapshot)

    with open(final_config_path, "w") as f:
        yaml.dump(frozen, f, sort_keys=False)

