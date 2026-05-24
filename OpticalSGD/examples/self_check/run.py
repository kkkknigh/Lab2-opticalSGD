from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_sgd.configuration.loader import load_config
from optical_sgd.experiments.experiment_setup import build_initial_patterns, build_renderer, build_scene
from optical_sgd.result_saving.image_saver import save_image
from optical_sgd.result_saving.output_directory import prepare_output_directory
from optical_sgd.result_saving.table_saver import save_metrics_json


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def run_self_check(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    scene = build_scene(config)
    renderer = build_renderer(config)

    metrics = {}
    for method in ["constant", "stripes", "random"]:
        method_cfg = {**config, "patterns": {**config["patterns"], "initial_method": method}}
        patterns = build_initial_patterns(method_cfg)
        result = renderer.render(patterns, scene)
        save_image(output_dir / f"{method}_pattern.png", patterns, cmap="gray")
        save_image(output_dir / f"{method}_captured_0.png", result.captured_images[0], cmap="gray")
        metrics[f"{method}_mean_intensity"] = float(result.captured_images.mean())

    save_image(output_dir / "ground_truth_correspondence.png", scene.correspondence)
    save_image(output_dir / "depth.png", scene.depth)
    save_image(output_dir / "albedo.png", scene.albedo, cmap="gray")
    save_image(output_dir / "specular.png", scene.specular, cmap="gray")
    save_image(output_dir / "scattering.png", scene.scattering, cmap="gray")
    metrics["material"] = scene.material_name
    metrics["material_description"] = scene.material_description
    save_metrics_json(output_dir / "metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_self_check(loaded.data, loaded.output_dir)
