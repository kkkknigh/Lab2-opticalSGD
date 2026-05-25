from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.common_logging import StepTimer, configure_log_file, log_step
from examples.train_patterns.run import run_pattern_training
from optical_sgd.configuration.loader import load_config
from optical_sgd.result_saving.savers import prepare_output_directory, save_metrics_json, save_rows_csv


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_frequency_comparison(config: dict, output_dir: Path) -> dict:
    """比较不同低通比例对 pattern 频谱、精度和运行时间的影响。"""

    output_dir = prepare_output_directory(output_dir)
    configure_log_file(output_dir / "run.log")
    log_step(f"frequency comparison output={output_dir}")
    fractions = config.get("frequency_comparison", {}).get("lowpass_fractions", [0.35])
    rows = []
    summary = {}
    for fraction in fractions:
        fraction = float(fraction)
        run_cfg = deepcopy(config)
        run_cfg["patterns"]["lowpass_fraction"] = fraction
        relative_dir = f"lowpass_{fraction:.2f}"
        with StepTimer(f"lowpass_fraction={fraction:.2f}"):
            metrics = run_pattern_training(run_cfg, output_dir / relative_dir, configure_logging=False)
        row = {"lowpass_fraction": fraction, **metrics}
        rows.append(row)
        summary[f"{fraction:.2f}"] = row
    save_rows_csv(output_dir / "frequency_constraint_comparison.csv", rows)
    save_metrics_json(output_dir / "metrics.json", summary)
    return summary


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_frequency_comparison(loaded.data, loaded.output_dir)
