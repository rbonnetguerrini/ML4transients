import sys
import yaml
from ML4transients.data_preparation.cutouts import save_cutouts

def main():
    # Use the first argument as config path, or default to configs/configs_cutout.yaml
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/configs_cutout.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    save_cutouts(config)

if __name__ == "__main__":
    main()