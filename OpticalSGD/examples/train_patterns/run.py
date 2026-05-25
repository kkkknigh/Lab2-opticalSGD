from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_sgd.evaluation.correspondence_metrics import error_map, threshold_accuracy
from optical_sgd.configuration.loader import load_config
from optical_sgd.experiments.experiment_setup import (
    build_decoder,
    build_initial_patterns,
    build_optimizer,
    build_renderer,
    build_scene,
)
from optical_sgd.pattern_generation.frequency_constraints import out_of_band_energy_ratio, spectrum_magnitude
from optical_sgd.result_saving.savers import (
    prepare_output_directory,
    save_checkpoint,
    save_image,
    save_line_plot,
    save_metrics_json,
)


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_pattern_training(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    scene = build_scene(config)
    renderer = build_renderer(config)
    decoder = build_decoder(config)
    patterns = build_initial_patterns(config)
    optimizer = build_optimizer(config, renderer, decoder, scene)

    initial_eval = optimizer.evaluate(patterns)
    optimized_patterns, state = optimizer.train(patterns)
    final_eval = optimizer.evaluate(optimized_patterns)
    final_result = final_eval["render_result"]

    err = error_map(final_eval["predicted"], final_result.ground_truth_correspondence)
    metrics = {
        "initial_loss": float(initial_eval["loss"]),
        "initial_mae": float(initial_eval["mae"]),
        "final_loss": float(final_eval["loss"]),
        "final_mae": float(final_eval["mae"]),
        "mean_gradient_norm": float(np.mean(state.gradient_norms)) if state.gradient_norms else 0.0,
        "std_gradient_norm": float(np.std(state.gradient_norms)) if state.gradient_norms else 0.0,
        "mean_decoder_gradient_norm": float(np.mean(state.decoder_gradient_norms))
        if state.decoder_gradient_norms
        else 0.0,
        "initial_out_of_band_energy_ratio": out_of_band_energy_ratio(
            patterns,
            float(config["patterns"]["lowpass_fraction"]),
        ),
        "optimized_out_of_band_energy_ratio": out_of_band_energy_ratio(
            optimized_patterns,
            float(config["patterns"]["lowpass_fraction"]),
        ),
        "accuracy_error_le_1": threshold_accuracy(
            final_eval["predicted"],
            final_result.ground_truth_correspondence,
            final_result.valid_mask,
            1.0,
        ),
    }
    save_image(output_dir / "initial_patterns.png", patterns, cmap="gray")
    save_image(output_dir / "optimized_patterns.png", optimized_patterns, cmap="gray")
    save_image(output_dir / "initial_pattern_spectrum.png", spectrum_magnitude(patterns), cmap="magma")
    save_image(output_dir / "optimized_pattern_spectrum.png", spectrum_magnitude(optimized_patterns), cmap="magma")
    save_image(output_dir / "final_captured_0.png", final_result.captured_images[0], cmap="gray")
    save_image(output_dir / "final_error_map.png", err)
    save_line_plot(output_dir / "loss_curve.png", state.losses, "loss")
    save_line_plot(output_dir / "mae_curve.png", state.maes, "mae")
    save_line_plot(output_dir / "decoder_gradient_norm_curve.png", state.decoder_gradient_norms, "decoder grad norm")
    save_checkpoint(output_dir / "checkpoint.npz", patterns=optimized_patterns)
    save_metrics_json(output_dir / "metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_pattern_training(loaded.data, loaded.output_dir)
