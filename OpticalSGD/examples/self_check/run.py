from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.common_logging import StepTimer, configure_log_file, log_step
from optical_sgd.configuration.loader import load_config
from optical_sgd.experiments.experiment_setup import build_initial_patterns, build_renderer, build_scene
from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.rendering.renderer_protocol import DifferentiableRendererProtocol, RendererProtocol
from optical_sgd.result_saving.savers import prepare_output_directory, save_image, save_metrics_json


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def render_for_self_check(renderer, patterns, scene) -> RenderResult:
    """为自检脚本生成 NumPy 渲染结果。"""

    if isinstance(renderer, DifferentiableRendererProtocol):
        torch_result = renderer.render_torch(patterns, scene)
        return RenderResult(
            captured_images=torch_result["captured_images"].detach().cpu().numpy(),
            ground_truth_correspondence=scene.correspondence,
            valid_mask=scene.valid_mask,
            albedo=scene.albedo,
            depth=scene.depth,
        )
    if isinstance(renderer, RendererProtocol):
        return renderer.render(patterns, scene)
    raise TypeError("Renderer must provide render_torch() or render().")


def run_self_check(config: dict, output_dir: Path) -> dict:
    output_dir = prepare_output_directory(output_dir)
    configure_log_file(output_dir / "run.log")
    log_step(f"self check output={output_dir}")
    with StepTimer("build scene/renderer"):
        scene = build_scene(config)
        renderer = build_renderer(config)

    metrics = {}
    for method in ["constant", "stripes", "random"]:
        with StepTimer(f"render method={method}"):
            method_cfg = {**config, "patterns": {**config["patterns"], "initial_method": method}}
            patterns = build_initial_patterns(method_cfg)
            result = render_for_self_check(renderer, patterns, scene)
            save_image(output_dir / f"{method}_pattern.png", patterns, cmap="gray")
            save_image(output_dir / f"{method}_captured_0.png", result.captured_images[0], cmap="gray")
        metrics[f"{method}_mean_intensity"] = float(result.captured_images.mean())

    with StepTimer("save scene maps"):
        save_image(output_dir / "ground_truth_correspondence.png", scene.correspondence)
        save_image(output_dir / "depth.png", scene.depth)
        save_image(output_dir / "albedo.png", scene.albedo, cmap="gray")
        save_image(output_dir / "specular.png", scene.specular, cmap="gray")
        save_image(output_dir / "scattering.png", scene.scattering, cmap="gray")
        metrics["material"] = scene.material_name
        metrics["material_description"] = scene.material_description
        save_metrics_json(output_dir / "metrics.json", metrics)
    log_step(f"summary material={scene.material_name}")
    return metrics


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_self_check(loaded.data, loaded.output_dir)
