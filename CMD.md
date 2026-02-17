# CMD 命令详解

本文档对仓库中每一条可执行命令行给出详细说明，包含作用、执行前配置项、预期输出和常见问题。

## 1. 安装可选统计依赖

### 命令
```bash
pip install nni
```

### 详细描述
- 作用: 安装 `nni`，用于各测试脚本中的 `count_flops_params`，统计 FLOPs 和参数量。
- 何时需要执行: 当你希望看到 `FLOPs` 和 `Params` 输出时。
- 不安装的影响: 前向测试仍可运行，但会提示 `NNI not installed`，并跳过统计。
- 验证方式: 安装成功后再次运行任意脚本，不再出现 `NNI not installed` 提示。

## 2. EMCAD 模块测试

### 命令
```bash
python emcad_mscam_lgag_eucb_sh_cvpr2024.py
```

### 详细描述
- 作用: 测试 EMCAD 相关模块的前向形状和可选复杂度统计。
- 配置入口: 修改文件内 `module_name` 变量。
- 可选值:
  - `"mscb"`: Multi-scale Convolution Block
  - `"mscam"`: Multi-scale Convolutional Attention Module
  - `"lgag"`: Large-kernel Grouped Attention Gate
  - `"eucb"`: Efficient Up-Convolution Block
  - `"sh"`: Segmentation Head
- 预期输出:
  - `Forward Pass Success` 表示前向正常
  - 安装 `nni` 后会输出 `FLOPs` 和 `Params`
- 失败排查:
  - `Unknown module name` 说明 `module_name` 配置非法
  - 通道维不匹配通常是手改输入形状导致

## 3. FocalMod/FocalBlock 测试

### 命令
```bash
python focalmod_focalnet_neurips2022.py
```

### 详细描述
- 作用: 测试 Focal Modulation 模块及其残差封装块。
- 配置入口: 修改文件内 `TEST_BLOCK` 变量。
- 可选值:
  - `"focal_mod"`: 仅测试 `FocalModulation`
  - `"block"`: 测试 `FocalModulationBlock`
- 预期输出:
  - 输入/输出 shape
  - `Forward Pass Success` 成功提示
  - 可选 FLOPs/Params 统计
- 失败排查:
  - 若统计失败但前向成功，优先检查 `nni` 是否安装

## 4. GRN / 多尺度深度卷积模块测试

### 命令
```bash
python grn_convnextv2_cvpr2023__msdw_mrlpfnet_iccv2023_emcad_cvpr2024.py
```

### 详细描述
- 作用: 测试 `GRN` 或 `MultiScaleDWConvBlock`。
- 配置入口: 修改文件内 `module_name` 变量。
- 可选值:
  - `"grn"`: Global Response Normalization
  - `"msdw"`: Multi-scale Depthwise Convolution Block
- 预期输出:
  - 前向 shape 检查结果
  - 可选 FLOPs/Params 输出
- 失败排查:
  - `Unknown module name` 说明测试名写错

## 5. MRLPF 系列模块测试

### 命令
```bash
python mrlpf_residual_lowpass_block_iccv2023.py
```

### 详细描述
- 作用: 测试 MRLPF 相关空间分支、频域分支、低通注意力和组合模块。
- 配置入口: 修改文件内 `module_name` 变量。
- 可选值:
  - `"spatial"`: `MRLPF_SpatialResidualBlock`
  - `"freq"`: `MRLPF_FrequencyResidualBlock`
  - `"lpf"`: `MRLPF_LowPassAttention`
  - `"mrlpf"`: `MRLPFBlock`
  - `"msrlpf"`: `MultiScaleResidualLowPassFilterBlock`
- 预期输出:
  - 前向成功信息和输出 shape
  - 可选 FLOPs/Params 统计
- 失败排查:
  - `Unknown module name` 为测试名拼写问题
  - 头数/通道相关报错通常由手工改配置导致

## 6. SFIM 系列模块测试

### 命令
```bash
python sfim_spatial_frequency_interaction_module_udc2025.py
```

### 详细描述
- 作用: 测试 SFIM 空间分支、频域分支、融合单元和完整模块。
- 配置入口: 修改文件内 `module_name` 变量。
- 可选值:
  - `"spatial"`: `SFIM_SpatialBranch`
  - `"freq"`: `SFIM_FrequencyBranch`
  - `"fuse"`: `SFIM_AttentionFusion`
  - `"sfim"`: `SFIMBlock`
- 预期输出:
  - 输入 shape 与前向成功提示
  - 可选 FLOPs/Params 输出
- 失败排查:
  - 融合模式 `fuse` 需要双输入，若改动脚本需保持输入数量一致

## 7. EDiT 模块测试

### 命令
```bash
python edit_diffusion_transformer_2025.py
```

### 详细描述
- 作用: 测试 `EDiTBlock` 或堆叠后的 `EDiT` 主干。
- 配置入口: 修改文件内 `module_name` 变量。
- 可选值:
  - `"edit_block"`: 单个 DiT 风格块
  - `"edit"`: 多层 EDiT 主干
- 预期输出:
  - 输入 shape、前向结果
  - 可选 FLOPs/Params 统计
- 失败排查:
  - Cross-attention 报错通常与 `context` 维度不一致有关

## 8. 执行建议

### 命令
```bash
python <script_name>.py
```

### 详细描述
- 推荐顺序: 先跑前向检查，再按需安装 `nni` 看复杂度统计。
- 统一成功标准: 出现 `Forward Pass Success` 即表示当前配置可用。
- 若仅做模块可用性验证，可忽略 FLOPs 统计相关报错。
