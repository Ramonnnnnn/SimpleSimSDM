import os
import yaml
from pathlib import Path

def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    simulation = cfg["simulation"]
    log_cfg = cfg["logging"]
    paths = cfg["paths"]
    rl = cfg["rl"]


    matrix_rows = simulation["matrix_rows"]
    matrix_cols = simulation["matrix_cols"]
    max_attempts = simulation["max_attempts"]
    rounds_per_load = simulation["rounds_per_load"]
    verbose = simulation["verbose"]
    seed = simulation["seed"]
    starting_load = simulation["starting_load"]
    final_load = simulation["final_load"]
    step = simulation["step"]
    use_multi_criteria = simulation["use_multi_criteria"]
    consider_crosstalk_threshold = simulation["consider_crosstalk_threshold"]
    region_finding_algorithm = simulation["region_finding_algorithm"]

    # RL stuff.
    rl_environment = simulation["rl_environment"]
    trained_model_path = rl["trained_model_path"]
    max_episode_length = rl["max_episode_length"]
    total_timesteps = rl["total_timesteps"]

    # paths
    base_dir = Path(__file__).parent

    log_name = log_cfg["log_name"]
    if "_" in log_name or ".csv" in log_name:
        raise KeyError("Can't put _ or .csv on log name")

    csv_files = [
        f"{prefix}_{log_name}.csv"
        for prefix in log_cfg["csv_prefixes"]
    ]

    csv_save_folder = base_dir / paths["csv_save_subfolder"]
    xml_path = base_dir / paths["xml_path"]

    # Flat namespace for retrieving when we run the simulation
    return {
        "matrix_rows": matrix_rows,
        "matrix_cols": matrix_cols,
        "max_attempts": max_attempts,
        "rounds_per_load": rounds_per_load,
        "verbose": verbose,
        "seed": seed,
        "starting_load": starting_load,
        "final_load": final_load,
        "step": step,
        "use_multi_criteria": use_multi_criteria,
        "consider_crosstalk_threshold": consider_crosstalk_threshold,
        "region_finding_algorithm": region_finding_algorithm,
        "rl_environment": rl_environment,
        "trained_model_path": trained_model_path,
        "max_episode_length": max_episode_length,
        "total_timesteps": total_timesteps,
        "base_dir": base_dir,
        "log_name": log_name,
        "csv_files": csv_files,
        "csv_save_folder": csv_save_folder,
        "xml_path": xml_path,
    }
