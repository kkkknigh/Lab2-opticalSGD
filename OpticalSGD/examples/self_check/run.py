from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.common_logging import StepTimer, configure_log_file, log_step
from optical_sgd.configuration.loader import load_config
from optical_sgd.experiments.experiment_setup import build_initial_patterns, build_renderer, build_scene
from optical_sgd.rendering.render_result import RenderResult
from optical_sgd.rendering.renderer_protocol import DifferentiableRendererProtocol, RendererProtocol
from optical_sgd.result_saving.savers import prepare_output_directory, save_image, save_metrics_json, save_rows_csv


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
    self_check_cfg = config.get("self_check", {})
    materials = self_check_cfg.get("materials", [config["scene"].get("material", "diffuse")])
    depth_profiles = self_check_cfg.get("depth_profiles", [config["scene"].get("depth_profile", "slanted_wave")])
    pattern_methods = self_check_cfg.get("pattern_methods", ["constant", "stripes", "random"])
    captured_indices = [int(index) for index in self_check_cfg.get("captured_indices", [0])]

    rows = []
    summary = {}
    for material in materials:
        summary[str(material)] = {}
        for depth_profile in depth_profiles:
            run_cfg = deepcopy(config)
            run_cfg["scene"]["material"] = str(material)
            run_cfg["scene"]["depth_profile"] = str(depth_profile)
            scene_dir = output_dir / str(material) / str(depth_profile)
            scene_dir = prepare_output_directory(scene_dir)
            with StepTimer(f"build scene/renderer material={material} depth={depth_profile}"):
                scene = build_scene(run_cfg)
                renderer = build_renderer(run_cfg)

            metrics = {
                "material": scene.material_name,
                "depth_profile": str(depth_profile),
                "material_description": scene.material_description,
            }
            for method in pattern_methods:
                method = str(method)
                with StepTimer(f"render material={material} depth={depth_profile} method={method}"):
                    method_cfg = deepcopy(run_cfg)
                    method_cfg["patterns"]["initial_method"] = method
                    patterns = build_initial_patterns(method_cfg)
                    result = render_for_self_check(renderer, patterns, scene)
                    save_image(scene_dir / f"{method}_pattern.png", patterns, cmap="gray")
                    for image_index in captured_indices:
                        if image_index < 0 or image_index >= result.captured_images.shape[0]:
                            raise ValueError(f"captured index {image_index} is outside rendered pattern count")
                        save_image(
                            scene_dir / f"{method}_captured_{image_index}.png",
                            result.captured_images[image_index],
                            cmap="gray",
                        )
                metrics[f"{method}_mean_intensity"] = float(result.captured_images.mean())

            with StepTimer(f"save scene maps material={material} depth={depth_profile}"):
                save_image(scene_dir / "ground_truth_correspondence.png", scene.correspondence)
                save_image(scene_dir / "depth.png", scene.depth)
                save_image(scene_dir / "albedo.png", scene.albedo, cmap="gray")
                save_image(scene_dir / "specular.png", scene.specular, cmap="gray")
                save_image(scene_dir / "scattering.png", scene.scattering, cmap="gray")
                save_metrics_json(scene_dir / "metrics.json", metrics)

            rows.append(metrics)
            summary[str(material)][str(depth_profile)] = metrics

    save_rows_csv(output_dir / "self_check_summary.csv", rows)
    save_metrics_json(output_dir / "metrics.json", summary)
    log_step(f"summary scenes={len(rows)}")
    return summary


if __name__ == "__main__":
    loaded = load_config(CONFIG_PATH)
    run_self_check(loaded.data, loaded.output_dir)
