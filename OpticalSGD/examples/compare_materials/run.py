from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.train_patterns.run import run_pattern_training
from optical_sgd.configuration.loader import load_config
from optical_sgd.result_saving.output_directory import prepare_output_directory
from optical_sgd.result_saving.table_saver import save_metrics_json, save_rows_csv


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_material_comparison(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    materials = config.get("material_comparison", {}).get("materials", ["diffuse", "marble", "wood", "frosted_glass"])
    rows = []
    summary = {}
    for material in materials:
        material_cfg = deepcopy(config)
        material_cfg["scene"]["material"] = material
        metrics = run_pattern_training(material_cfg, output_dir / material)
        row = {"material": material, **metrics}
        rows.append(row)
        summary[material] = row
    save_rows_csv(output_dir / "material_comparison.csv", rows)
    save_metrics_json(output_dir / "metrics.json", summary)
    return summary


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_material_comparison(loaded.data, loaded.output_dir)
