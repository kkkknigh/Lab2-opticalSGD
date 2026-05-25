from pathlib import Path
import sys
from copy import deepcopy
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.common_logging import StepTimer, configure_log_file, log_step
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
    configure_log_file(output_dir / "run.log")
    log_step(f"gradient comparison output={output_dir}")
    comparison_cfg = config.get("gradient_comparison", {})
    stability_seeds = comparison_cfg.get("stability_seeds", [int(config["patterns"]["seed"])])
    with StepTimer("gradient stability estimate"):
        stability_rows = [
            _estimate_gradient_stability(_with_seed(config, seed), seed=seed)
            for seed in stability_seeds
        ]
    save_rows_csv(output_dir / "gradient_stability.csv", stability_rows)

    rows = []
    summary = {}
    for method in ["finite_difference", "autograd"]:
        method_cfg = deepcopy(config)
        method_cfg["optimization"]["gradient_method"] = method
        with StepTimer(f"main gradient_method={method}"):
            start = perf_counter()
            metrics = run_pattern_training(method_cfg, output_dir / method, configure_logging=False)
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
        with StepTimer(f"epsilon sensitivity epsilon={float(epsilon):.4f}"):
            start = perf_counter()
            metrics = run_pattern_training(
                epsilon_cfg,
                output_dir / f"epsilon_{float(epsilon):.4f}",
                configure_logging=False,
            )
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
            with StepTimer(f"noise sensitivity method={method} noise={float(noise_std):.3f}"):
                start = perf_counter()
                metrics = run_pattern_training(
                    noise_cfg,
                    output_dir / f"noise_{method}_{float(noise_std):.3f}",
                    configure_logging=False,
                )
            noise_rows.append(
                {
                    "gradient_method": method,
                    "noise_std": float(noise_std),
                    "runtime_seconds": perf_counter() - start,
                    **metrics,
                }
            )
    save_rows_csv(output_dir / "noise_sensitivity.csv", noise_rows)

    pattern_count_rows = []
    pattern_counts = comparison_cfg.get("pattern_counts", [])
    for method in ["finite_difference", "autograd"]:
        for pattern_count in pattern_counts:
            sample_cfg = deepcopy(config)
            sample_cfg["optimization"]["gradient_method"] = method
            sample_cfg["patterns"]["count"] = int(pattern_count)
            sample_cfg["optimization"]["iterations"] = int(
                comparison_cfg.get("sensitivity_iterations", config["optimization"]["iterations"])
            )
            with StepTimer(f"pattern-count sensitivity method={method} count={int(pattern_count)}"):
                start = perf_counter()
                metrics = run_pattern_training(
                    sample_cfg,
                    output_dir / f"patterns_{method}_{int(pattern_count)}",
                    configure_logging=False,
                )
            pattern_count_rows.append(
                {
                    "gradient_method": method,
                    "pattern_count": int(pattern_count),
                    "runtime_seconds": perf_counter() - start,
                    **metrics,
                }
            )
    save_rows_csv(output_dir / "pattern_count_sensitivity.csv", pattern_count_rows)

    seed_rows = []
    seeds = comparison_cfg.get("seeds", [])
    for seed in seeds:
        for method in ["finite_difference", "autograd"]:
            seed_cfg = _with_seed(config, seed)
            seed_cfg["optimization"]["gradient_method"] = method
            seed_cfg["optimization"]["iterations"] = int(
                comparison_cfg.get("repeat_iterations", config["optimization"]["iterations"])
            )
            with StepTimer(f"seed repeat method={method} seed={int(seed)}"):
                start = perf_counter()
                metrics = run_pattern_training(
                    seed_cfg,
                    output_dir / f"seed_{int(seed)}_{method}",
                    configure_logging=False,
                )
            seed_rows.append(
                {
                    "seed": int(seed),
                    "gradient_method": method,
                    "runtime_seconds": perf_counter() - start,
                    **metrics,
                }
            )
    save_rows_csv(output_dir / "seed_repeats.csv", seed_rows)
    seed_summary_rows = _summarize_by_method(seed_rows)
    save_rows_csv(output_dir / "seed_repeat_summary.csv", seed_summary_rows)

    full_summary = {
        "gradient_stability": stability_rows,
        "main_comparison": summary,
        "finite_difference_epsilon_sensitivity": epsilon_rows,
        "noise_sensitivity": noise_rows,
        "pattern_count_sensitivity": pattern_count_rows,
        "seed_repeats": seed_rows,
        "seed_repeat_summary": seed_summary_rows,
    }
    save_metrics_json(output_dir / "metrics.json", full_summary)
    return full_summary


def _estimate_gradient_stability(config: dict, seed: int | None = None) -> dict:
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
        "seed": int(config["patterns"]["seed"] if seed is None else seed),
        "epsilon": float(config["optimization"]["epsilon"]),
        "pattern_count": int(config["patterns"]["count"]),
        "finite_difference_gradient_norm": fd_norm,
        "autograd_gradient_norm": autograd_norm,
        "gradient_norm_ratio_fd_over_autograd": fd_norm / max(autograd_norm, 1e-12),
        "gradient_difference_norm": difference_norm,
        "gradient_relative_difference": difference_norm / max(autograd_norm, 1e-12),
        "gradient_cosine_similarity": cosine_similarity(fd_gradient, autograd_gradient),
        "finite_difference_runtime_seconds": fd_seconds,
        "autograd_runtime_seconds": autograd_seconds,
    }


def _with_seed(config: dict, seed: int) -> dict:
    seeded = deepcopy(config)
    seeded["patterns"]["seed"] = int(seed)
    return seeded


def _summarize_by_method(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    metric_names = [
        "final_mae",
        "accuracy_error_le_1",
        "runtime_seconds",
        "mean_gradient_norm",
        "std_gradient_norm",
        "mae_reaches_final_10pct_iteration",
    ]
    methods = sorted({str(row["gradient_method"]) for row in rows})
    summaries = []
    for method in methods:
        method_rows = [row for row in rows if str(row["gradient_method"]) == method]
        summary = {"gradient_method": method, "runs": len(method_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in method_rows if metric in row]
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_gradient_comparison(loaded.data, loaded.output_dir)
