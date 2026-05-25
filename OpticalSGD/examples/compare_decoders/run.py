from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.common_logging import StepTimer, log_step
from examples.train_patterns.run import run_pattern_training
from optical_sgd.configuration.loader import load_config
from optical_sgd.result_saving.savers import prepare_output_directory, save_metrics_json, save_rows_csv


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_decoder_comparison(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    log_step(f"decoder comparison output={output_dir}")
    comparison_cfg = config.get("decoder_comparison", {})
    variants = comparison_cfg.get("variants")
    if variants is None:
        decoder_types = comparison_cfg.get("types", ["zncc", "zncc_neighborhood", "zncc_nn"])
        variants = [
            {
                "name": decoder_type,
                "type": "zncc" if decoder_type == "zncc_neighborhood" else decoder_type,
                "neighborhood": 3 if decoder_type in {"zncc_neighborhood", "zncc_nn"} else 1,
            }
            for decoder_type in decoder_types
        ]
    materials = comparison_cfg.get("materials", [config["scene"].get("material", "diffuse")])
    rows = []
    summary = {}
    for material in materials:
        summary[material] = {}
        for variant in variants:
            decoder_name = str(variant["name"])
            decoder_cfg = deepcopy(config)
            decoder_cfg["scene"]["material"] = material
            decoder_cfg["decoder"]["type"] = str(variant["type"])
            decoder_cfg["decoder"]["neighborhood"] = int(variant.get("neighborhood", 1))
            decoder_cfg["optimization"]["joint_optimize_decoder"] = bool(
                variant.get("joint_optimize_decoder", decoder_cfg["optimization"].get("joint_optimize_decoder", False))
            )
            with StepTimer(f"decoder={decoder_name} material={material}"):
                metrics = run_pattern_training(decoder_cfg, output_dir / material / decoder_name)
            row = {
                "material": material,
                "decoder": decoder_name,
                "decoder_type": str(variant["type"]),
                "neighborhood": int(variant.get("neighborhood", 1)),
                "joint_optimize_decoder": bool(decoder_cfg["optimization"]["joint_optimize_decoder"]),
                **metrics,
            }
            rows.append(row)
            summary[material][decoder_name] = row
    save_rows_csv(output_dir / "decoder_comparison.csv", rows)
    save_metrics_json(output_dir / "metrics.json", summary)
    return summary


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_decoder_comparison(loaded.data, loaded.output_dir)
