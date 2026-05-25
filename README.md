# OpticalSGD Lab

## 环境依赖

```bash
pip install -r requirements.txt
```

## 单元测试
测试主要检查单个函数的输入输出。

```bash
pytest OpticalSGD/tests
pytest OpticalSGD/tests/test_synthetic_scene.py
pytest OpticalSGD/tests/test_synthetic_scene.py::test_create_scene_returns_scene_description_with_expected_shapes
```

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
        |   |-- decoder_protocol.py                             # DecoderProtocol、TorchFeatureTransformProtocol 等抽象接口
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

## 关键实现

### 渲染后端

`renderer.backend: torch` 使用 PyTorch 实现可微结构光渲染链路：

```text
patterns -> captured_images -> decoder scores -> loss -> pattern gradients
```

Torch 渲染器提供 `render_torch()` 接口，返回 tensor 结果并保留梯度路径。

`renderer.backend: mitsuba` 使用 Mitsuba 3，将一维 pattern 扩展成 projector emitter 的二维 bitmap texture，再由 camera sensor 渲染观测图。该后端是黑盒物理渲染器，只提供 `render()`，用于有限差分和物理渲染对比，不支持 autograd。

### 材质和场景

合成场景使用相机视角下的深度图表示几何，使用材质贴图表示外观。每个材质包含：

- `albedo`：漫反射强度。
- `specular`：镜面反射强度。
- `scattering`：相邻投影列混合强度。
- `projector_gamma` / `camera_gamma`：设备亮度响应。

当前材质包括：

- `diffuse`：理想漫反射基准材质，纹理弱、无明显高光和散射。
- `marble`：大理石材质，包含高对比石纹、局部镜面高光和轻微散射。
- `wood`：木纹材质，包含横向纹理和年轮式亮暗变化，用来测试重复纹理的影响。
- `frosted_glass`：磨砂玻璃近似材质，散射更强、对比度更低，用来测试半透明模糊外观下的鲁棒性。

### Decoder

`zncc` (Zero-mean Normalized Cross-Correlation)，为零均值归一化互相关。`decoder.neighborhood: n` 表示 ZNCC 使用的横向局部窗口宽度。

`zncc_nn` 在 ZNCC 前加入可学习模块：

- projector 端 32-bin 分段响应曲线，模拟 projector/camera 的非线性亮度响应。
- camera feature residual MLP。
- projector feature residual MLP。

decoder 参数更新：
- `optimization.joint_optimize_decoder: false` 时，只更新 pattern。
- `optimization.joint_optimize_decoder: true` 时，decoder 参数和 pattern 分开更新。
- 如果 decoder 支持 autograd 参数接口，decoder 参数直接用 Torch 反传更新，不取决于 pattern 梯度是 `autograd` 还是 `finite_difference`。

autograd：
- pattern 梯度来自 Torch 计算图，不能调用 NumPy 版 decoder。
- 如果 decoder 实现 `transform_torch_features()`，优化器会在 Torch 特征上调用这个接口。
- 如果 `optimization.joint_optimize_decoder: true` 且 decoder 实现 autograd 参数接口，decoder 参数会以 `requires_grad=True` 的 Torch 张量参与 loss 反传。
- 当前 `zncc_nn` 的响应曲线和 residual MLP 支持这种真正的 autograd 更新；普通 `zncc` 没有可学习参数。
- pattern 和 decoder 是两组参数，分别使用 `learning_rate` 和 `decoder_learning_rate`；pattern 更新后还会做亮度裁剪和频率约束，decoder 参数不会做 pattern 的低通约束。

### 梯度方式

`gradient_method: autograd` ：使用 PyTorch 直接反传 loss 到 pattern；如果开启 `joint_optimize_decoder`，同时反传并更新支持 autograd 的 decoder 参数。

`gradient_method: finite_difference` 使用链式估计：

1. 渲染当前 pattern 得到 baseline captured images。
2. 对 captured images 计算 `d loss / d image`。
3. 对每个 pattern 控制量做 plus/minus 扰动并重新渲染。
4. 估计 `d image / d pattern`，得到光学路径梯度。
5. 额外加入 decoder codebook 对 pattern 的直接依赖项。

### 频率约束

每次 pattern 更新后都会执行：

- `clamp_patterns()`：把亮度限制到 `[0, 1]`。
- `apply_frequency_constraint()`：沿 projector 宽度做 FFT 低通。

训练会保存 `initial_pattern_spectrum.png`、`optimized_pattern_spectrum.png` 和带外能量占比，用于说明优化后 pattern 是否满足频率上限。

## 实验入口

每个实验都有自己的配置文件，路径 `OpticalSGD/examples/<experiment>/config.yaml`。

### 渲染器自检

关注点：检查常量图案、条纹图案、随机图案的渲染结果，保存深度、albedo、specular、scattering 和 correspondence 真值图，证明投影方向和几何对应关系合理。

配置：

```text
OpticalSGD/examples/self_check/config.yaml
```

运行：

```bash
python OpticalSGD/examples/self_check/run.py
```

### Pattern 训练

关注点：从初始 pattern 出发运行 OpticalSGD，比较优化前后 loss、MAE、频谱、误差图和相机观测图。

配置：

```text
OpticalSGD/examples/train_patterns/config.yaml
```

运行：

```bash
python OpticalSGD/examples/train_patterns/run.py
```

### 梯度方式对比

关注点：固定场景和 decoder，对比 `finite_difference` 与 `autograd` 的收敛结果、梯度方向相似度、扰动步长敏感性和噪声敏感性。

配置：

```text
OpticalSGD/examples/compare_gradients/config.yaml
```

运行：

```bash
python OpticalSGD/examples/compare_gradients/run.py
```

### Decoder 对比

关注点：比较 `zncc`、带邻域的 `zncc` 和 `zncc_nn` 在相同材质/场景下的 loss、MAE 和稳定性。该实验固定 renderer、材质、pattern 初始化和梯度方式，只改变 decoder variant；其中 `zncc_nn` 额外开启 `joint_optimize_decoder`，因此它代表“可学习 decoder”方案，不应和固定参数 ZNCC 当作完全同参量消融。

配置：

```text
OpticalSGD/examples/compare_decoders/config.yaml
```

运行：

```bash
python OpticalSGD/examples/compare_decoders/run.py
```

### 材质对比

关注点：比较 `diffuse`、`marble`、`wood`、`frosted_glass` 对优化结果和解码误差的影响；其中 `marble` 对应作业要求的大理石材质实验。

配置：

```text
OpticalSGD/examples/compare_materials/config.yaml
```

运行：

```bash
python OpticalSGD/examples/compare_materials/run.py
```

### 渲染器对比

关注点：固定有限差分路径，对比 Torch 近似渲染器和 Mitsuba 物理渲染器；同时输出 `torch/autograd` 与 `mitsuba/finite_difference` 的系统路径对比。

配置：

```text
OpticalSGD/examples/compare_renderers/config.yaml
```

运行：

```bash
python OpticalSGD/examples/compare_renderers/run.py
```

## 输出内容

常见输出文件：

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

对比实验还会输出 CSV：

- `gradient_comparison.csv`
- `gradient_stability.csv`
- `finite_difference_epsilon_sensitivity.csv`
- `noise_sensitivity.csv`
- `decoder_comparison.csv`
- `material_comparison.csv`
- `renderer_comparison.csv`
- `system_comparison.csv`
