from pathlib import Path
import sys
from copy import deepcopy
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.train_patterns.run import run_pattern_training
from optical_sgd.configuration.loader import load_config
from optical_sgd.evaluation.gradient_metrics import cosine_similarity
from optical_sgd.experiments.experiment_setup import (
    build_decoder,
    build_initial_patterns,
    build_optimizer,
    build_renderer,
    build_scene,
)
from optical_sgd.optimization.gradient_estimators import AutogradGradientEstimator
from optical_sgd.pattern_generation.frequency_constraints import apply_frequency_constraint, clamp_patterns
from optical_sgd.result_saving.savers import prepare_output_directory, save_metrics_json, save_rows_csv


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_gradient_comparison(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    stability_rows = [_estimate_gradient_stability(config)]
    save_rows_csv(output_dir / "gradient_stability.csv", stability_rows)

    rows = []
    summary = {}
    for method in ["finite_difference", "autograd"]:
        method_cfg = deepcopy(config)
        method_cfg["optimization"]["gradient_method"] = method
        start = perf_counter()
        metrics = run_pattern_training(method_cfg, output_dir / method)
        elapsed = perf_counter() - start
        row = {"gradient_method": method, "runtime_seconds": elapsed, **metrics}
        rows.append(row)
        summary[method] = row
    save_rows_csv(output_dir / "gradient_comparison.csv", rows)

    epsilon_rows = []
    base_epsilon = float(config["optimization"]["epsilon"])
    epsilons = config.get("gradient_comparison", {}).get(
        "epsilons",
        [base_epsilon * 0.5, base_epsilon, base_epsilon * 2.0],
    )
    for epsilon in epsilons:
        epsilon_cfg = deepcopy(config)
        epsilon_cfg["optimization"]["gradient_method"] = "finite_difference"
        epsilon_cfg["optimization"]["epsilon"] = float(epsilon)
        epsilon_cfg["optimization"]["iterations"] = int(
            config.get("gradient_comparison", {}).get(
                "sensitivity_iterations",
                config["optimization"]["iterations"],
            )
        )
        start = perf_counter()
        metrics = run_pattern_training(epsilon_cfg, output_dir / f"epsilon_{float(epsilon):.4f}")
        epsilon_rows.append({"epsilon": float(epsilon), "runtime_seconds": perf_counter() - start, **metrics})
    save_rows_csv(output_dir / "finite_difference_epsilon_sensitivity.csv", epsilon_rows)

    noise_rows = []
    base_noise = float(config["renderer"]["noise_std"])
    noise_values = config.get("gradient_comparison", {}).get("noise_levels", [base_noise, max(base_noise, 0.01), 0.03])
    for method in ["finite_difference", "autograd"]:
        for noise_std in noise_values:
            noise_cfg = deepcopy(config)
            noise_cfg["optimization"]["gradient_method"] = method
            noise_cfg["renderer"]["noise_std"] = float(noise_std)
            noise_cfg["optimization"]["iterations"] = int(
                config.get("gradient_comparison", {}).get(
                    "sensitivity_iterations",
                    config["optimization"]["iterations"],
                )
            )
            start = perf_counter()
            metrics = run_pattern_training(noise_cfg, output_dir / f"noise_{method}_{float(noise_std):.3f}")
            noise_rows.append(
                {
                    "gradient_method": method,
                    "noise_std": float(noise_std),
                    "runtime_seconds": perf_counter() - start,
                    **metrics,
                }
            )
    save_rows_csv(output_dir / "noise_sensitivity.csv", noise_rows)

    full_summary = {
        "gradient_stability": stability_rows,
        "main_comparison": summary,
        "finite_difference_epsilon_sensitivity": epsilon_rows,
        "noise_sensitivity": noise_rows,
    }
    save_metrics_json(output_dir / "metrics.json", full_summary)
    return full_summary


def _estimate_gradient_stability(config: dict) -> dict:
    scene = build_scene(config)
    patterns = build_initial_patterns(config)
    patterns = apply_frequency_constraint(patterns, float(config["patterns"]["lowpass_fraction"]))

    def make_optimizer(method: str):
        method_cfg = deepcopy(config)
        method_cfg["optimization"]["gradient_method"] = method
        method_cfg["optimization"]["iterations"] = 1
        return build_optimizer(method_cfg, build_renderer(method_cfg), build_decoder(method_cfg), scene)

    fd_optimizer = make_optimizer("finite_difference")
    autograd_optimizer = make_optimizer("autograd")

    def loss_function(candidate: np.ndarray) -> float:
        constrained = apply_frequency_constraint(
            clamp_patterns(candidate),
            float(config["patterns"]["lowpass_fraction"]),
        )
        return float(fd_optimizer.evaluate(constrained)["loss"])

    fd_start = perf_counter()
    fd_gradient = fd_optimizer._optical_finite_difference_gradient(patterns, loss_function)
    fd_seconds = perf_counter() - fd_start

    autograd_start = perf_counter()
    autograd_gradient = AutogradGradientEstimator(float(config["optimization"]["epsilon"])).estimate(
        patterns,
        loss_function,
        autograd_optimizer._torch_soft_zncc_loss,
    )
    autograd_seconds = perf_counter() - autograd_start

    fd_norm = float(np.linalg.norm(fd_gradient))
    autograd_norm = float(np.linalg.norm(autograd_gradient))
    difference_norm = float(np.linalg.norm(fd_gradient - autograd_gradient))
    return {
        "epsilon": float(config["optimization"]["epsilon"]),
        "finite_difference_gradient_norm": fd_norm,
        "autograd_gradient_norm": autograd_norm,
        "gradient_norm_ratio_fd_over_autograd": fd_norm / max(autograd_norm, 1e-12),
        "gradient_difference_norm": difference_norm,
        "gradient_relative_difference": difference_norm / max(autograd_norm, 1e-12),
        "gradient_cosine_similarity": cosine_similarity(fd_gradient, autograd_gradient),
        "finite_difference_runtime_seconds": fd_seconds,
        "autograd_runtime_seconds": autograd_seconds,
    }


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_gradient_comparison(loaded.data, loaded.output_dir)
