import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Multi-scale Residual Low-Pass Filter Block (MRLPF Block)
#
# Paper:
#   "Multi-Scale Residual Low-Pass Filter Network for Image Deblurring"
#   (ICCV 2023, Dong et al.)
#
# 对应论文中的 RLPF 模块 (Eqs. (3), (4), (5)):
#
#   F = Y + R(Y) + F_freq(Y)                         (3)
#   H_i = sum_j S_ij f_j     (Self-Attention as LPF) (4)
#   E = F + D(H)                                    (5)
#
# 这里给出一个“忠实 + 工程友好”的 PyTorch 实现：
#   - MRLPF_SpatialResidualBlock:    R(Y)
#   - MRLPF_FrequencyResidualBlock:  F_freq(Y)
#   - MRLPF_LowPassAttention:        self-attention 低通 + depthwise conv
#   - MRLPFBlock / MultiScaleResidualLowPassFilterBlock: 组合成完整模块
#
# 输入 / 输出: (B, C, H, W) -> (B, C, H, W)
# ============================================================


# ------------------------------------------------------------
# 1. Spatial Residual Branch  R(Y)
#    两个 3x3 卷积 + ReLU，对应论文里的 Conv3 + ReLU + Conv3
# ------------------------------------------------------------

class MRLPF_SpatialResidualBlock(nn.Module):
    """
    Spatial Residual Block for MRLPF:

        R(Y) = Conv3x3(ReLU(Conv3x3(Y)))

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


# ------------------------------------------------------------
# 2. Frequency Residual Branch  F_freq(Y)
#    参考 DeepRFT / 论文 [15] 的做法：
#    - 2D FFT → 实部/虚部在通道维拼接
#    - Conv1x1 + ReLU + Conv1x1
#    - 再 iFFT 回空间域，得到 F_freq(Y)
# ------------------------------------------------------------

class MRLPF_FrequencyResidualBlock(nn.Module):
    """
    Frequency Residual Block for MRLPF:

        F_freq(Y) = iFFT( Conv1x1( ReLU( Conv1x1( FFT(Y) ) ) ) )

    这里 FFT 使用 torch.fft.fft2 / ifft2，
    实部和虚部在通道维拼接，用 Conv1x1 进行线性变换。

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

        # 2D FFT
        spec = torch.fft.fft2(x, norm="ortho")  # complex: (B, C, H, W)
        real = spec.real
        imag = spec.imag

        feat = torch.cat([real, imag], dim=1)  # (B, 2C, H, W)
        feat = self.act(self.conv1(feat))
        feat = self.conv2(feat)

        real2, imag2 = torch.chunk(feat, 2, dim=1)
        spec_new = torch.complex(real2, imag2)

        # iFFT 回到空间域，取实部
        out = torch.fft.ifft2(spec_new, norm="ortho").real  # (B, C, H, W)
        return out


# ------------------------------------------------------------
# 3. Learnable Low-Pass Filter via Self-Attention
#    MRLPF_LowPassAttention 对应 Eqs. (4)-(5)：
#
#      H_i = sum_j S_ij f_j,   S = softmax(QK^T / sqrt(d))
#      E = F + D(H)
#
#    这里实现为标准 scaled dot-product self-attention +
#    一个 3x3 depthwise conv 作为 D(·)。
# ------------------------------------------------------------

class MRLPF_LowPassAttention(nn.Module):
    """
    Learnable Low-Pass Filter via Self-Attention.

    输入:  F ∈ R^{B×C×H×W}
    输出:  E ∈ R^{B×C×H×W}

    步骤:
      1. 把 F 视为 N=H*W 个 token，维度 C
      2. 基于 F 做 Q,K,V (线性层)，自注意力得到 H (低通特征)
      3. H reshape 回 (B,C,H,W)，过 depthwise Conv3x3 → D(H)
      4. E = F + D(H)
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # D(·): depthwise Conv3x3
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(channels)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F.shape
        N = H * W

        # reshape to (B, N, C)
        x = F.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Q,K,V: (B, N, C)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 多头拆分: (B, num_heads, N, head_dim)
        def split_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Attention: (B, num_heads, N, N)
        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        # H: (B, num_heads, N, head_dim)
        Hh = torch.matmul(attn, Vh)

        # 合并头: (B, N, C)
        H_flat = Hh.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        H_flat = self.out_proj(H_flat)  # (B, N, C)

        # reshape 回 (B, C, H, W)
        H_feat = H_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # D(H) depthwise conv
        DH = self.dw_bn(self.dw(H_feat))

        # 残差
        E = F + DH
        return E


# ------------------------------------------------------------
# 4. MRLPF Block (Residual Low-Pass Filter Block)
#    最终组合模块，对应论文里的 RLPF 模块。
#
#    F = Y + R(Y) + F_freq(Y)
#    E = LowPassAttention(F)
# ------------------------------------------------------------

class MRLPFBlock(nn.Module):
    """
    MRLPF Block (Residual Low-Pass Filter Block).

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)

    你可以把它直接当成:
      - ResNet Block 的替代
      - UNet bottleneck / encoder / decoder block 的替代
      - EIT / CT / Deblur 等重建网络里的“低通先验”模块
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.spatial_branch = MRLPF_SpatialResidualBlock(channels)
        self.freq_branch = MRLPF_FrequencyResidualBlock(channels)
        self.lpf = MRLPF_LowPassAttention(channels, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. (3): F = Y + R(Y) + F_freq(Y)
        spatial_res = self.spatial_branch(x)
        freq_res = self.freq_branch(x)
        F = x + spatial_res + freq_res

        # Eq. (4)(5): learnable low-pass filter via self-attention
        E = self.lpf(F)
        return E


# 兼容你在外面引用时喜欢用的长名字
class MultiScaleResidualLowPassFilterBlock(MRLPFBlock):
    """
    别名：MultiScaleResidualLowPassFilterBlock
    实际上就是 MRLPFBlock，本身不做多尺度下采样，
    “multi-scale” 是整个 MRLPFNet 框架的 coarse-to-fine 结构带来的。
    """
    pass


# ============================================================
# 5. 测试脚本
#    - Forward shape 检查
#    - NNI 统计 FLOPs / Params（和你 EcNet 小脚本同风格）
# ============================================================

def build_test_module(name: str):
    """
    支持测试:
      - "spatial"  : MRLPF_SpatialResidualBlock
      - "freq"     : MRLPF_FrequencyResidualBlock
      - "lpf"      : MRLPF_LowPassAttention
      - "mrlpf"    : MRLPFBlock
      - "msrlpf"   : MultiScaleResidualLowPassFilterBlock
    """
    name = name.lower()
    C = 32
    H = W = 32
    x = torch.rand(1, C, H, W)

    if name == "spatial":
        module = MRLPF_SpatialResidualBlock(channels=C)
        inputs = (x,)
    elif name == "freq":
        module = MRLPF_FrequencyResidualBlock(channels=C)
        inputs = (x,)
    elif name == "lpf":
        module = MRLPF_LowPassAttention(channels=C, num_heads=4)
        inputs = (x,)
    elif name == "mrlpf":
        module = MRLPFBlock(channels=C, num_heads=4)
        inputs = (x,)
    elif name == "msrlpf":
        module = MultiScaleResidualLowPassFilterBlock(channels=C, num_heads=4)
        inputs = (x,)
    else:
        raise ValueError(f"Unknown module name: {name}")

    return module, inputs


if __name__ == "__main__":
    # 这里改名字就能测不同模块:
    # "spatial", "freq", "lpf", "mrlpf", "msrlpf"
    module_name = "mrlpf"

    model, inputs = build_test_module(module_name)

    print(f"🔧 Testing MRLPF module: {module_name}")
    in_shapes = ", ".join(str(t.shape) for t in inputs)

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
