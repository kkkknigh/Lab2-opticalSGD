# OpticalSGD Lab 

## 环境依赖

```bash
pip install -r requirements.txt
```

## 单元测试

```bash
pytest tests
```

测试按模块拆分，覆盖配置读取、pattern 生成与频率约束、几何采样、decoder、指标、有限差分梯度和材质/深度，不运行完整训练实验。

## 代码框架

```text
OpticalSGD/
    |-- examples/                                               # 各实验的运行入口
    |   |-- self_check/
    |   |   |-- config.yaml                                     # 自检实验配置
    |   |   `-- run.py                                          # 组装场景、pattern、渲染自检并保存结果
    |   |-- train_patterns/
    |   |   |-- config.yaml                                     # pattern 训练配置
    |   |   `-- run.py                                          # 调用优化器训练并保存图像、曲线、checkpoint、metrics
    |   |-- compare_gradients/
    |   |   |-- config.yaml                                     # 梯度对比配置
    |   |   `-- run.py                                          # 对比 finite_difference/autograd 并保存 CSV
    |   |-- compare_decoders/
    |   |   |-- config.yaml                                     # decoder 对比配置
    |   |   `-- run.py                                          # 对比 zncc/zncc_neighborhood/zncc_nn
    |   |-- compare_materials/
    |   |   |-- config.yaml                                     # 材质对比配置
    |   |   `-- run.py                                          # 对比 diffuse/marble/wood/frosted_glass
    |   `-- compare_renderers/
    |       |-- config.yaml                                     # 渲染器对比配置
    |       `-- run.py                                          # 固定 finite_difference，对比 Torch 与 Mitsuba
    `-- optical_sgd/                                            # 主要代码
        |-- __init__.py                                         
        |-- configuration/                                      # YAML 配置读取、合并、基础校验
        |   |-- default.yaml                                    # renderer/scene/pattern/decoder/optimization 默认值
        |   |-- loader.py                                       # 读取 YAML、加载 default.yaml、合并实验配置
        |   `-- schema.py                                       # ExperimentConfig 配置类与 validate_config 配置基础校验
        |-- experiments/                                        # 实验准备工具，不放具体实验流程
        |   `-- experiment_setup.py                             # build_scene/renderer/decoder/patterns/optimizer
        |-- rendering/                                          # projector-camera 几何和渲染后端
        |   |-- render_result.py                                # RenderResult 数据类，保存 captured_images/correspondence/valid_mask
        |   |-- renderer_protocol.py                            # 黑盒 render() 与可微 render_torch() 协议
        |   |-- projector_camera_model.py                       # 生成 camera 到 projector 的 correspondence 并采样 projector 列
        |   |-- torch_renderer.py                               # PyTorch 可微渲染器，提供 render_torch()
        |   `-- mitsuba_renderer.py                             # Mitsuba 3 物理渲染器，作为黑盒有限差分后端
        |-- synthetic_scene/                                    # 合成场景、深度面、材质贴图和几何真值
        |   |-- __init__.py                                     # 对外导出 synthetic_scene 公共 API
        |   |-- scene.py                                        # SceneDescription/create_scene/depth
        |   `-- materials/                                      # 结构化材质参数模块
        |       |-- __init__.py                                 # 注册 make_material_maps() 和各材质构造函数
        |       |-- base.py                                     # MaterialMaps 数据类、normalized_grid()、constant_map()
        |       |-- diffuse.py                                  # make_diffuse()，Lambertian 漫反射基准材质
        |       |-- marble.py                                   # make_marble()，大理石纹理、veins、高光和轻微散射
        |       |-- wood.py                                     # make_wood()，木纹/年轮纹理材质
        |       `-- frosted_glass.py                            # make_frosted_glass()，强局部散射的磨砂玻璃近似
        |-- pattern_generation/                                 # pattern 初始化、约束和频谱统计
        |   |-- initial_patterns.py                             # create_initial_patterns()，生成 random/stripes/constant pattern
        |   `-- frequency_constraints.py                        # clamp、FFT 低通、频谱幅值和带外能量占比
        |-- correspondence_decoding/                            # 从相机观测恢复 projector 列对应关系
        |   |-- decoder_protocol.py                             # DecoderProtocol、TrainableDecoderProtocol 等抽象接口
        |   |-- feature_extraction.py                           # 特征归一化、projector/camera 邻域特征构造
        |   |-- zncc_decoder.py                                 # DecoderOutput 与可配置邻域的 ZNCCDecoder
        |   `-- zncc_neural_decoder.py                          # ZNCCNeuralDecoder，响应曲线与 residual MLP 可学习 decoder
        |-- optimization/                                       # OpticalSGD 主循环、loss 和梯度估计
        |   |-- optimizer_state.py                              # OptimizerState，记录 iteration/loss/MAE/梯度范数历史
        |   |-- correspondence_losses.py                        # correspondence_mae() 与 soft_expected_l1_loss()
        |   |-- gradient_estimators.py                          # 有限差分和 Torch autograd 梯度估计器
        |   `-- optical_sgd_optimizer.py                        # OpticalSGDOptimizer，更新 patterns 并可联合更新 decoder 参数
        |-- evaluation/                                         # 实验指标和计时工具
        |   |-- correspondence_metrics.py                       # threshold_accuracy() 与 error_map()
        |   |-- gradient_metrics.py                             # cosine_similarity()，比较梯度方向
        |   `-- runtime_metrics.py                              # measure_seconds() 上下文管理器，记录运行耗时
        `-- result_saving/                                      # 统一保存实验输出
            `-- savers.py                                       # 输出目录、图片、曲线、JSON/CSV 和 checkpoint 保存
```

## 模块职责与接口

### `configuration/`

负责读取 `OpticalSGD/optical_sgd/configuration/default.yaml`，再用实验 YAML 覆盖默认值，并做基础字段校验。

对外接口：
- `load_config(config_path) -> ExperimentConfig`：读取、合并、校验配置。
- `ExperimentConfig`：保存 `data` 和 `config_path`，提供 `experiment_name`、`output_dir` 属性。
- `validate_config(data)`：检查必需配置段和基础数值范围。

### `experiments/`

只保留实验准备工具，不承载具体训练、对比、保存图片/CSV/checkpoint 的实验流程。具体流程放在 `examples/<experiment>/run.py`。

对外接口：
- `build_scene(config)`：按配置创建合成场景。
- `build_renderer(config)`：按配置创建 Torch 或 Mitsuba 渲染器。
- `build_decoder(config)`：按配置创建 ZNCC 或 ZNCC-NN decoder；ZNCC 的邻域大小由 `decoder.neighborhood` 控制。
- `build_initial_patterns(config)`：按配置创建初始 patterns。
- `build_optimizer(config, renderer, decoder, scene)`：按配置创建 `OpticalSGDOptimizer`。

### `rendering/`

实现 projector-camera 几何、渲染结果结构和两类渲染后端。Torch 后端支持 autograd，Mitsuba 后端作为物理黑盒用于有限差分。

对外接口：
- `TorchRenderer(config)`：PyTorch 可微渲染器，提供 `render_torch()`。
- `MitsubaRenderer(config)`：Mitsuba 物理渲染器，提供 `render()`。
- `RendererProtocol`：黑盒渲染协议，约定 `render(patterns, scene) -> RenderResult`。
- `DifferentiableRendererProtocol`：可微渲染协议，约定 `render_torch(patterns, scene)`。
- `RenderResult`：保存 `captured_images`、`correspondence`、`valid_mask` 等渲染输出。
- `make_correspondence_map(...)`：生成 camera 到 projector 的几何对应关系和有效 mask。
- `sample_projector_columns(patterns, columns)`：按对应列采样 projector pattern。

### `synthetic_scene/`

生成合成场景，包括深度面、材质贴图、真实 correspondence 和有效 mask。

对外接口：
- `create_scene(config) -> SceneDescription`：从配置创建完整场景。
- `SceneDescription`：保存 `depth`、`correspondence`、`valid_mask`、`material_maps` 等场景数据。
- `make_depth_surface(height, width, profile)`：生成 `flat`、`bump`、`slanted_wave` 等深度面。
- `make_material_maps(height, width, material) -> MaterialMaps`：生成结构化材质参数。
- `MaterialMaps`：保存 `albedo`、`specular`、`scattering`、`projector_gamma`、`camera_gamma`。

### `pattern_generation/`

负责 pattern 初始化、取值约束、频率约束和频谱统计。

对外接口：
- `create_initial_patterns(count, width, method, seed)`：生成 `random`、`stripes`、`constant` 等初始 pattern。
- `clamp_patterns(patterns)`：把 pattern 限制到 `[0, 1]`。
- `apply_frequency_constraint(patterns, lowpass_fraction)`：FFT 低通约束。
- `spectrum_magnitude(patterns)`：计算频谱幅值。
- `out_of_band_energy_ratio(patterns, lowpass_fraction)`：统计带外能量占比。

### `correspondence_decoding/`

从相机观测图中估计每个 camera pixel 对应的 projector 列。

对外接口：
- `DecoderProtocol`：decoder 基础协议，约定 `feature_radius` 和 `decode(captured_images, patterns)`。
- `TrainableDecoderProtocol`：可训练 decoder 协议，约定 `parameter_vector()` 和 `set_parameter_vector()`。
- `TorchFeatureTransformProtocol`：可微 decoder 特征变换协议，用于 autograd 路径。
- `ZNCCDecoder`：ZNCC 匹配 decoder；`neighborhood: 1` 是逐像素匹配，`neighborhood > 1` 时加入同一行局部邻域特征。
- `ZNCCNeuralDecoder`：带可学习响应曲线和 residual MLP 的 decoder。
- `DecoderOutput`：保存 `correspondence`、`confidence`、`score_volume` 等输出。
- `normalize_features(features)`：特征归一化。
- `projector_neighborhood_features(patterns, radius)`：构造 projector 邻域特征。
- `camera_neighborhood_features(images, radius)`：构造 camera 邻域特征。

### `optimization/`

实现 OpticalSGD 主优化循环、两种梯度估计方式和 correspondence loss。

对外接口：
- `OpticalSGDOptimizer`：主优化器，依赖渲染器协议、`DecoderProtocol` 和 `SceneDescription`，负责更新 patterns，可选联合更新 decoder。
- `OptimizerState`：保存迭代数、loss、MAE、梯度范数等历史状态。
- `FiniteDifferenceGradientEstimator`：黑盒有限差分估计 image-Jacobian 或完整 loss 梯度。
- `AutogradGradientEstimator`：基于 Torch autograd 的梯度估计。
- `correspondence_mae(predicted, ground_truth, valid_mask)`：计算有效区域 MAE。
- `soft_expected_l1_loss(...)`：基于 soft matching 分布的可微 L1 loss。

### `evaluation/`

提供实验指标和简单计时工具。

对外接口：
- `threshold_accuracy(predicted, ground_truth, valid_mask, threshold)`：阈值准确率。
- `error_map(predicted, ground_truth)`：逐像素误差图。
- `cosine_similarity(a, b)`：梯度方向相似度。
- `measure_seconds(container, key)`：上下文管理器，把耗时写入字典。

### `result_saving/`

统一保存实验输出，包括图片、曲线、JSON、CSV 和 checkpoint。

对外接口：
- `prepare_output_directory(path) -> Path`：创建输出目录。
- `save_image(path, image, cmap="viridis")`：保存单张图像。
- `save_line_plot(path, values, ylabel)`：保存曲线图。
- `save_metrics_json(path, metrics)`：保存 metrics。
- `save_rows_csv(path, rows)`：保存表格。
- `save_checkpoint(path, **arrays)`：保存 `.npz` checkpoint。

### `examples/`

每个子目录包含一个具体实验的 `config.yaml` 和 `run.py`。这里负责组装训练或对比流程、收集 metrics、保存图片、曲线、CSV 和 checkpoint。

对外入口：
- `python OpticalSGD/examples/self_check/run.py`
- `python OpticalSGD/examples/train_patterns/run.py`
- `python OpticalSGD/examples/compare_gradients/run.py`
- `python OpticalSGD/examples/compare_decoders/run.py`
- `python OpticalSGD/examples/compare_materials/run.py`
- `python OpticalSGD/examples/compare_renderers/run.py`

## 怎么运行

```bash
python OpticalSGD/examples/self_check/run.py
python OpticalSGD/examples/train_patterns/run.py
python OpticalSGD/examples/compare_gradients/run.py
python OpticalSGD/examples/compare_decoders/run.py
python OpticalSGD/examples/compare_materials/run.py
python OpticalSGD/examples/compare_renderers/run.py
```

## 实现说明

### 依赖倒置

核心优化器不直接创建具体的渲染器、decoder 类。`examples/` 层根据配置调用 `build_scene()`、`build_renderer()`、`build_decoder()`、`build_initial_patterns()` 和 `build_optimizer()` 组装实验。

`OpticalSGDOptimizer` 只依赖抽象接口：

- `RendererProtocol`：黑盒渲染器提供 `render()`。
- `DifferentiableRendererProtocol`：可微渲染器提供 `render_torch()`。
- `DecoderProtocol`：提供 `feature_radius` 和 `decode()`。
- `TrainableDecoderProtocol`：当 decoder 可学习时提供参数向量读写。
- `TorchFeatureTransformProtocol`：当 decoder 有可微特征变换时提供 torch 特征变换。

优化器按协议选择渲染路径：实现 `render_torch()` 的可微渲染器走 Torch 张量路径，实现 `render()` 的黑盒渲染器走 NumPy `RenderResult` 路径。新增 renderer 或 decoder 时，实现相应协议并在 `experiments/experiment_setup.py` 注册构建逻辑即可。

### 渲染器

`renderer.backend: torch` 使用 PyTorch 实现可微结构光渲染链路，支持 `patterns -> captured_images -> loss` 的 autograd。渲染流程是：

1. 根据几何真值 `scene.correspondence` 对 projector pattern 做线性采样。
2. 应用材质的 projector gamma。
3. 用材质 `scattering` 做局部列混合，近似间接光/半透明扩散。
4. 叠加深度 shading、albedo、specular highlight、ambient。
5. 应用 camera gamma 和噪声。

`renderer.backend: mitsuba` 使用官方 Mitsuba 3 包，不再是 NumPy 仿真实现。它通过 Mitsuba 的 projector emitter 将每张 1D pattern 转成 2D bitmap 投影到物理场景中，再用 camera sensor 渲染观测图。该后端用于黑盒有限差分，不提供 `render_torch()`，因此不能用于 `gradient_method: autograd`。

Mitsuba 后端需要安装：

```bash
pip install mitsuba
```

配置示例：

```yaml
renderer:
  backend: mitsuba
  mitsuba_variant: auto   # auto 优先 cuda_ad_rgb，其次 llvm_ad_rgb，最后 scalar_rgb
  spp: 16
  camera_fov: 42.0
  projector_fov: 38.0
  projector_scale: 6.0
optimization:
  gradient_method: finite_difference
```

本项目将变量拆开对比：

- `compare_gradients` 固定 `torch` 渲染器，对比 `finite_difference` 与 `autograd`。
- `compare_renderers` 固定 `finite_difference`，对比 `torch` 与 `mitsuba` 渲染器。
- `compare_renderers` 还会额外输出系统路径综合对比：`torch/autograd` 与 `mitsuba/finite_difference`，该结果不作为单变量消融结论。

对应入口是：

```bash
python OpticalSGD/examples/compare_renderers/run.py
```

GPU 设置写在配置的 `renderer.device` 字段：

```yaml
renderer:
  backend: torch
  device: auto   # auto 会优先用 cuda；也可以显式写 cuda 或 cpu
```

### 材质

材质统一放在 `OpticalSGD/optical_sgd/synthetic_scene/materials/`。每个材质模块返回同一张“表结构”：

- `albedo`：表面反射强度。
- `specular`：镜面高光强度。
- `scattering`：局部列混合强度，用来近似间接光、半透明和模糊传输。
- `projector_gamma` / `camera_gamma`：设备非线性响应。

当前包含：

- `diffuse`：Lambertian 基准。
- `marble`：大理石纹理，包含高对比 veins、轻微 specular 和 scattering。
- `wood`：木纹和年轮纹理。
- `frosted_glass`：磨砂玻璃近似，重点模拟强局部散射。

### Decoder

`zncc` 使用同一个 `ZNCCDecoder` 实现；`decoder.neighborhood: 1` 是逐像素 K 维强度向量匹配，`decoder.neighborhood > 1` 时将特征扩展到同一行的 `1 x p` 邻域。  
`zncc_nn` 不再是固定非线性占位；它包含：

- 投影端 32-bin 分段响应曲线 `g()`。
- camera feature residual MLP。
- projector feature residual MLP。
- 变换后再做 ZNCC 匹配。

当 decoder 提供 `parameter_vector()` / `set_parameter_vector()` 时，`OpticalSGDOptimizer` 会用同一个 correspondence loss 对 decoder 参数做有限差分更新，从而实现 patterns 和 decoder 的联合优化。

当 decoder 提供 `transform_torch_features()` 时，autograd 路径会调用该接口做可微特征变换；优化器不会按具体 decoder 类名分支。

### 梯度

`gradient_method: autograd` 通过 `TorchRenderer.render_torch()` 直接反传 loss 到 patterns。

`gradient_method: finite_difference` 尽量贴近论文 OpticalSGD 的链式做法：

1. 渲染当前 pattern 得到 baseline captured images。
2. 对 captured images 计算 `d loss / d image`。
3. 对每个 pattern 控制量做 plus/minus 扰动并重新渲染，估计 image Jacobian。
4. 用 `d loss / d image * d image / d pattern` 得到光学路径梯度。
5. 额外加入 decoder codebook 对 pattern 的直接依赖项。

如果当前后端无法提供 image-loss 梯度，代码会退回到完整 loss 的中心差分。

### 频率约束

每次更新后都会执行：

- `clamp_patterns()`：限制 pattern 到 `[0, 1]`。
- `apply_frequency_constraint()`：沿 projector 宽度做 FFT 低通。

训练输出会额外保存：

- `initial_pattern_spectrum.png`
- `optimized_pattern_spectrum.png`
- `initial_out_of_band_energy_ratio`
- `optimized_out_of_band_energy_ratio`

用于说明优化后 pattern 是否满足预设频率上限。

## 输出内容

常见输出文件包括：

- `metrics.json`
- `initial_patterns.png`
- `optimized_patterns.png`
- `initial_pattern_spectrum.png`
- `optimized_pattern_spectrum.png`
- `final_captured_0.png`
- `final_error_map.png`
- `loss_curve.png`
- `mae_curve.png`
- `decoder_gradient_norm_curve.png`
- `checkpoint.npz`

`compare_gradients` 还会输出：

- `gradient_comparison.csv`
- `gradient_stability.csv`
- `finite_difference_epsilon_sensitivity.csv`
- `noise_sensitivity.csv`

`compare_decoders` 会在 `diffuse`、`marble`、`frosted_glass` 等材质下比较：

- `zncc`
- `zncc_neighborhood`
- `zncc_nn`

并输出：

- `decoder_comparison.csv`

`compare_renderers` 会在同一组材质、噪声和几何配置下比较：

- `torch/finite_difference`
- `mitsuba/finite_difference`

并额外做综合路径对比：

- `torch/autograd`
- `mitsuba/finite_difference`

并输出：

- `renderer_comparison.csv`
- `system_comparison.csv`
