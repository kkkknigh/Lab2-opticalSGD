from pathlib import Path
import sys
from copy import deepcopy
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.train_patterns.run import run_pattern_training
from optical_sgd.configuration.loader import load_config
from optical_sgd.result_saving.savers import prepare_output_directory, save_metrics_json, save_rows_csv


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_renderer_comparison(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    analysis_cfg = config.get("renderer_comparison", {})
    materials = analysis_cfg.get("materials", ["diffuse", "marble", "wood", "frosted_glass"])
    depth_profiles = analysis_cfg.get("depth_profiles", ["flat", "bump", "slanted_wave"])
    noise_values = analysis_cfg.get("noise_levels", [float(config["renderer"]["noise_std"])])
    backends = analysis_cfg.get("backends", ["torch", "mitsuba"])
    gradient_method = str(analysis_cfg.get("gradient_method", "finite_difference"))
    rows = []
    for backend in backends:
        for material in materials:
            for depth_profile in depth_profiles:
                for noise_std in noise_values:
                    run_cfg = deepcopy(config)
                    run_cfg["renderer"]["backend"] = backend
                    run_cfg["renderer"]["noise_std"] = float(noise_std)
                    run_cfg["scene"]["material"] = material
                    run_cfg["scene"]["depth_profile"] = depth_profile
                    run_cfg["optimization"]["gradient_method"] = gradient_method
                    relative_dir = f"{backend}_{gradient_method}_{material}_{depth_profile}_noise{float(noise_std):.3f}"
                    start = perf_counter()
                    metrics = run_pattern_training(run_cfg, output_dir / relative_dir)
                    rows.append(
                        {
                            "backend": backend,
                            "gradient_method": gradient_method,
                            "material": material,
                            "depth_profile": depth_profile,
                            "noise_std": float(noise_std),
                            "runtime_seconds": perf_counter() - start,
                            **metrics,
                        }
                    )

    system_paths = analysis_cfg.get(
        "system_paths",
        [
            {"name": "torch_autograd", "backend": "torch", "gradient_method": "autograd"},
            {"name": "mitsuba_finite_difference", "backend": "mitsuba", "gradient_method": "finite_difference"},
        ],
    )
    system_rows = []
    for path_cfg in system_paths:
        path_name = str(path_cfg["name"])
        backend = str(path_cfg["backend"])
        path_gradient_method = str(path_cfg["gradient_method"])
        for material in materials:
            for depth_profile in depth_profiles:
                for noise_std in noise_values:
                    run_cfg = deepcopy(config)
                    run_cfg["renderer"]["backend"] = backend
                    run_cfg["renderer"]["noise_std"] = float(noise_std)
                    run_cfg["scene"]["material"] = material
                    run_cfg["scene"]["depth_profile"] = depth_profile
                    run_cfg["optimization"]["gradient_method"] = path_gradient_method
                    relative_dir = f"system_{path_name}_{material}_{depth_profile}_noise{float(noise_std):.3f}"
                    start = perf_counter()
                    metrics = run_pattern_training(run_cfg, output_dir / relative_dir)
                    system_rows.append(
                        {
                            "path": path_name,
                            "backend": backend,
                            "gradient_method": path_gradient_method,
                            "material": material,
                            "depth_profile": depth_profile,
                            "noise_std": float(noise_std),
                            "runtime_seconds": perf_counter() - start,
                            **metrics,
                        }
                    )

    summary = {"renderer_controlled_comparison": rows, "system_comparison": system_rows}
    save_rows_csv(output_dir / "renderer_comparison.csv", rows)
    save_rows_csv(output_dir / "system_comparison.csv", system_rows)
    save_metrics_json(output_dir / "metrics.json", summary)
    return summary


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_renderer_comparison(loaded.data, loaded.output_dir)
