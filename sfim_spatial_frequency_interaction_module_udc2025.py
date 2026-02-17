import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SFIM: Spatial–Frequency Interaction Module (通用版本)
#
# 思路来源：
#   - "Integrating Spatial and Frequency Information for
#      Under-Display Camera Image Restoration" (SFIM, UDC, 2025)
#     中的 SDB/FDB/AMIB 思路：
#       * 空间分支: CNN 捕捉局部细节 (噪声 & 模糊)
#       * 频域分支: FFT + 频域网络捕捉全局结构 (flare 等)
#       * 注意力式融合: 自适应融合空间 & 频域特征
#
#   - 以及 FSI (ICCV'23) 的 frequency–spatial 双分支交互。
#
# 这里实现一个简化但工程友好的版本：
#   - SFIM_SpatialBranch       : 空间卷积分支
#   - SFIM_FrequencyBranch     : 频域滤波分支 (FFT / iFFT)
#   - SFIM_AttentionFusion     : 空间-频域注意力融合
#   - SFIMBlock / SpatialFrequencyInteractionModule :
#        一个即插即用的 Conv + FFT 混合块
#
# 输入 / 输出: (B, C, H, W) -> (B, C, H, W)
# ============================================================


# ------------------------------------------------------------
# 1. 空间分支: SDB 风格 (简单残差卷积块)
# ------------------------------------------------------------

class SFIM_SpatialBranch(nn.Module):
    """
    Spatial Branch (SDB-like).

    非常标准的残差卷积块：
        R_s(x) = Conv3x3(BN + ReLU(Conv3x3(x)))

    Args:
        channels: 输入/输出通道数 C
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # 残差回加
        out = self.act(x + y)
        return out


# ------------------------------------------------------------
# 2. 频域分支: FDB 风格 (FFT + 1x1 conv 滤波 + iFFT)
# ------------------------------------------------------------

class SFIM_FrequencyBranch(nn.Module):
    """
    Frequency Branch (FDB-like).

    简化实现：
      - 对输入做 2D FFT
      - 把实部 / 虚部在通道维拼接
      - 频域上用 1x1 Conv + ReLU + 1x1 Conv 做线性变换
      - 再 iFFT 回到空间域

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels

        # 2D FFT: 得到 complex 特征
        spec = torch.fft.fft2(x, norm="ortho")  # (B, C, H, W), complex
        real = spec.real
        imag = spec.imag

        # 在通道维拼接实部和虚部
        feat = torch.cat([real, imag], dim=1)  # (B, 2C, H, W)
        feat = self.act(self.conv1(feat))
        feat = self.conv2(feat)

        # 再拆回实部 / 虚部
        real2, imag2 = torch.chunk(feat, 2, dim=1)
        spec_new = torch.complex(real2, imag2)

        # iFFT 回到空间域，取实部
        x_rec = torch.fft.ifft2(spec_new, norm="ortho").real  # (B, C, H, W)
        return x_rec


# ------------------------------------------------------------
# 3. 空间-频域注意力融合单元
#    类似 AMIB 的局部版本：
#      - concat(Fs, Ff) -> Conv1x1 -> 2C 通道的 gate
#      - 拆成 g_s, g_f ∈ (0,1)，分别调制 Fs, Ff
#      - fused = g_s * Fs + g_f * Ff
# ------------------------------------------------------------

class SFIM_AttentionFusion(nn.Module):
    """
    Attention-based Spatial–Frequency Fusion.

    Inputs:
        Fs: 空间分支输出 (B, C, H, W)
        Ff: 频域分支输出 (B, C, H, W)

    Output:
        fused: (B, C, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.fuse_conv = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=2 * channels,
            kernel_size=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(2 * channels)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fs: torch.Tensor, Ff: torch.Tensor) -> torch.Tensor:
        B, C, H, W = Fs.shape
        assert Ff.shape == Fs.shape

        joint = torch.cat([Fs, Ff], dim=1)  # (B, 2C, H, W)
        gate = self.sigmoid(self.bn(self.fuse_conv(joint)))  # (B, 2C, H, W)

        g_s, g_f = torch.chunk(gate, 2, dim=1)  # (B, C, H, W) each

        Fs_mod = Fs * g_s
        Ff_mod = Ff * g_f

        fused = Fs_mod + Ff_mod  # (B, C, H, W)
        return fused


# ------------------------------------------------------------
# 4. 整体 SFIM Block：Conv + FFT + 交互
# ------------------------------------------------------------

class SFIMBlock(nn.Module):
    """
    Spatial–Frequency Interaction Module (SFIM Block).

    Pipeline:
      1) Fs = SpatialBranch(x)
      2) Ff = FrequencyBranch(x)
      3) F_fused = AttentionFusion(Fs, Ff)
      4) out = x + Conv1x1(BN + ReLU(F_fused))

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)

    用法示例:
      - 直接替换 UNet / EcNet encoder 的某个残差块:
            self.block3 = SFIMBlock(channels=64)
      - 或作为 bottleneck 前后的“去模糊 + 去噪声”专用模块。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.spatial = SFIM_SpatialBranch(channels)
        self.freq = SFIM_FrequencyBranch(channels)
        self.fusion = SFIM_AttentionFusion(channels)

        # 输出投影 + 残差
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Fs = self.spatial(x)
        Ff = self.freq(x)
        fused = self.fusion(Fs, Ff)       # (B, C, H, W)

        y = self.out_bn(self.out_conv(fused))
        y = self.out_act(y)

        out = x + y
        return out


# 给一个别名，方便你在别的文件里用全名调用
class SpatialFrequencyInteractionModule(SFIMBlock):
    """
    别名：SpatialFrequencyInteractionModule
    实际上就是 SFIMBlock。
    """
    pass


# ============================================================
# 5. 测试脚本
#    - Forward shape 检查
#    - NNI 统计 FLOPs / Params
# ============================================================

def build_test_module(name: str):
    """
    支持测试:
      - "spatial" : SFIM_SpatialBranch
      - "freq"    : SFIM_FrequencyBranch
      - "fuse"    : SFIM_AttentionFusion
      - "sfim"    : SFIMBlock / SpatialFrequencyInteractionModule
    """
    name = name.lower()
    C = 32
    H = W = 32

    x = torch.rand(1, C, H, W)
    if name == "spatial":
        module = SFIM_SpatialBranch(channels=C)
        inputs = (x,)
    elif name == "freq":
        module = SFIM_FrequencyBranch(channels=C)
        inputs = (x,)
    elif name == "fuse":
        module = SFIM_AttentionFusion(channels=C)
        Fs = torch.rand(1, C, H, W)
        Ff = torch.rand(1, C, H, W)
        inputs = (Fs, Ff)
    elif name == "sfim":
        module = SFIMBlock(channels=C)
        inputs = (x,)
    else:
        raise ValueError(f"Unknown module name: {name}")

    return module, inputs


if __name__ == "__main__":
    # 这里改名字就能测不同模块:
    # "spatial", "freq", "fuse", "sfim"
    module_name = "sfim"

    model, inputs = build_test_module(module_name)
    in_shapes = ", ".join(str(t.shape) for t in inputs)

    print(f"🔧 Testing SFIM module: {module_name}")
    print(f"   Input shape(s): {in_shapes}")

    # --- Forward 测试 ---
    try:
        with torch.no_grad():
            out = model(*inputs)

        if isinstance(out, tuple):
            out_shapes = ", ".join(str(t.shape) for t in out)
        else:
            out_shapes = str(out.shape)

        print(f"✅ Forward Pass Success: {in_shapes} → {out_shapes}")
    except Exception as e:
        print(f"❌ Forward Failed: {e}")

    # --- FLOPs / Params ---
    try:
        from nni.compression.utils.counter import count_flops_params

        flops, params, _ = count_flops_params(model, x=inputs)
        print(f"📊 FLOPs:  {flops / 1e6:.2f} MFLOPs | Params: {params / 1e6:.4f} M")
    except ImportError:
        print("⚠️ NNI not installed. Run: pip install nni")
    except Exception as e:
        print(f"⚠️ FLOPs/Params counting failed: {e}")
