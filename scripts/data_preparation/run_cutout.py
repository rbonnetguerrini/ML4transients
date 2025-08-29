import sys
import yaml
from pathlib import Path
from ML4transients.data_preparation.cutouts import save_cutouts
from ML4transients.data_preparation.lightcurves import extract_and_save_lightcurves_with_index
from datetime import datetime
from ML4transients.utils import realtime_update, append_config

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data_preparation/configs_cutout.yaml"
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    realtime_update(config_path, "running")

    try:
        save_cutouts(config)
        
        # Extract lightcurves with cross-reference indices
        if config.get("extract_lightcurves", True):  # Add this flag to config
            print("\n=== Extracting lightcurves and creating indices ===")
            extract_and_save_lightcurves_with_index(config)
        
        config["run_info"]["status"] = "completed"
    except Exception as e:
        config["run_info"]["status"] = "failed"
        raise
    finally:
        config["run_info"]["finished"] = datetime.utcnow().isoformat()
        append_config(config)
        realtime_update(config_path, config["run_info"]["status"])

if __name__ == "__main__":
    main()