import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EMCAD: Efficient Multi-scale Convolutional Attention Decoding
#   CVPR 2024
#   Paper: "EMCAD: Efficient Multi-scale Convolutional Attention Decoding
#           for Medical Image Segmentation"
#   Official repo: https://github.com/SLDGroup/EMCAD
#
# 这里整理的是几个“可抠出来单独用”的小模块（简化实现版本）：
#   - MSCB: Multi-scale Convolution Block
#   - MSCAM: Multi-scale Convolutional Attention Module
#   - LGAG: Large-kernel Grouped Attention Gate
#   - EUCB: Efficient Up-Convolution Block
#   - SegHead: 1x1 segmentation head
#
# 全部都按 NCHW (B, C, H, W) 写，方便插到 UNet / EcNet / PVT 等结构里。
# ============================================================


# -----------------------------
# 小工具: channel shuffle
# -----------------------------
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    标准的 channel shuffle 操作：
    把通道分组后重新打乱，增强组间信息交互。
    """
    b, c, h, w = x.size()
    assert c % groups == 0, "channels must be divisible by groups"
    x = x.view(b, groups, c // groups, h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(b, c, h, w)
    return x


# ============================================================
# 1. Multi-scale Convolution Block (MSCB)
#    - 参考 EMCAD 论文中 MSCB 的设计思路:
#      inverted residual (扩张 -> 多尺度 depthwise -> 压缩 + 通道 shuffle)
# ============================================================

class EMCAD_MSCB(nn.Module):
    """
    Multi-scale Convolution Block (简化版).

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)
    - 先用 1x1 卷积扩展通道 (factor=2)
    - 多尺度 depthwise conv 顺序堆叠，并用残差形式累计
    - 通道 shuffle 促进组间信息交互
    - 1x1 卷积压回原始通道数，并加上 input 残差
    """

    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        kernel_sizes=(3, 5, 7),
        shuffle_groups: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.expanded_dim = dim * expansion
        self.kernel_sizes = kernel_sizes
        self.shuffle_groups = shuffle_groups

        # 1x1 conv: 扩展通道
        self.expand = nn.Sequential(
            nn.Conv2d(dim, self.expanded_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expanded_dim),
            nn.ReLU6(inplace=True),
        )

        # 多尺度 depthwise conv 序列
        dw_layers = []
        for k in kernel_sizes:
            p = k // 2
            dw_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.expanded_dim,
                        self.expanded_dim,
                        kernel_size=k,
                        padding=p,
                        groups=self.expanded_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expanded_dim),
                    nn.ReLU6(inplace=True),
                )
            )
        self.ms_dw = nn.ModuleList(dw_layers)

        # 压回原始通道
        self.project = nn.Sequential(
            nn.Conv2d(self.expanded_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        identity = x

        out = self.expand(x)  # (B, 2C, H, W)

        # 顺序堆叠多尺度 depthwise conv，每层都有残差
        for dw in self.ms_dw:
            out = out + dw(out)

        # 通道 shuffle
        if self.shuffle_groups > 1:
            out = channel_shuffle(out, self.shuffle_groups)

        # 压回原通道并加上 identity
        out = self.project(out)
        if out.shape == identity.shape:
            out = out + identity

        out = self.act(out)
        return out


# ============================================================
# 2. Channel Attention Block (CAB) & Spatial Attention Block (SAB)
#    - 使用类似 SE / CBAM 的经典实现，论文中也是通道+空间注意力组合
# ============================================================

class EMCAD_CAB(nn.Module):
    """
    Channel Attention Block (CAB) - 类似 SE, 用 GAP + MLP 做通道注意力。
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.mlp(w)
        return x * w


class EMCAD_SAB(nn.Module):
    """
    Spatial Attention Block (SAB) - 类似 CBAM 的空间注意力，
    用 avg_pool + max_pool 的拼接再做 7x7 conv。
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道维做 avg / max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s


# ============================================================
# 3. Multi-scale Convolutional Attention Module (MSCAM)
#    - CAB + SAB + MSCB
# ============================================================

class EMCAD_MSCAM(nn.Module):
    """
    Multi-scale Convolutional Attention Module.

    输入 / 输出: (B, C, H, W) -> (B, C, H, W)
    MSCAM(x) = MSCB(SAB(CAB(x)))，外面再加一次 residual。
    """

    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        kernel_sizes=(3, 5, 7),
        shuffle_groups: int = 4,
        reduction: int = 16,
    ):
        super().__init__()
        self.cab = EMCAD_CAB(dim, reduction=reduction)
        self.sab = EMCAD_SAB(kernel_size=7)
        self.mscb = EMCAD_MSCB(
            dim=dim,
            expansion=expansion,
            kernel_sizes=kernel_sizes,
            shuffle_groups=shuffle_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.cab(x)
        out = self.sab(out)
        out = self.mscb(out)
        # 再加一层短残差，略微贴近论文“refinement”的感觉
        if out.shape == identity.shape:
            out = out + identity
        return out


# ============================================================
# 4. Large-kernel Grouped Attention Gate (LGAG)
#    - 来自 EMCAD 解码器中的大核分组注意力门：
#      用 3x3 group conv 分别处理 g (skip) 和 x (up-sampled)，
#      再 1x1 conv + Sigmoid 得到单通道 gate，对 x 做缩放。
# ============================================================

class EMCAD_LGAG(nn.Module):
    """
    Large-kernel Grouped Attention Gate (LGAG).

    Args:
        channels: g 和 x 的通道数 (假设已经 match 好)
        groups:   group conv 的组数

    Inputs:
        g: 来自 skip connection 的特征 (B, C, H, W)
        x: 上采样后的当前 stage 特征 (B, C, H, W)

    Output:
        gated x: (B, C, H, W)
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.gc_g = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.gc_x = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.bn_g = nn.BatchNorm2d(channels)
        self.bn_x = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.bn_out = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g, x : (B, C, H, W)，假设已经同分辨率
        g_feat = self.bn_g(self.gc_g(g))
        x_feat = self.bn_x(self.gc_x(x))

        h = self.relu(g_feat + x_feat)
        att = self.conv1x1(h)
        att = self.bn_out(att)
        att = self.sigmoid(att)  # (B, 1, H, W)

        out = x * att
        return out


# ============================================================
# 5. Efficient Up-Convolution Block (EUCB)
#    - 上采样模块: UpSampling -> depthwise 3x3 -> 1x1 conv
# ============================================================

class EMCAD_EUCB(nn.Module):
    """
    Efficient Up-Convolution Block (EUCB).

    输入 / 输出:
        输入: (B, C_in, H, W)
        输出: (B, C_out, 2H, 2W)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.ReLU(inplace=True)

        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_bn(self.pw(x))
        return x


# ============================================================
# 6. Segmentation Head (SH)
#    - 非常简单: 1x1 卷积把通道数映射到类别数
# ============================================================

class EMCAD_SegHead(nn.Module):
    """
    Segmentation Head (SH).

    输入: (B, C_in, H, W)
    输出: (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# 7. 测试脚本 (和你 EcNet 的风格保持一致)
#    - Forward shape 测试
#    - NNI 统计 FLOPs / Params (可选)
# ============================================================

def build_test_module(name: str):
    """
    根据名字构造一个待测试模块 + 输入张量。
    """
    name = name.lower()

    if name == "mscb":
        dim = 16
        module = EMCAD_MSCB(dim=dim)
        inputs = (torch.rand(1, dim, 32, 32),)  # x

    elif name == "mscam":
        dim = 16
        module = EMCAD_MSCAM(dim=dim)
        inputs = (torch.rand(1, dim, 32, 32),)  # x

    elif name == "lgag":
        dim = 16
        module = EMCAD_LGAG(channels=dim)
        g = torch.rand(1, dim, 32, 32)
        x = torch.rand(1, dim, 32, 32)
        inputs = (g, x)

    elif name == "eucb":
        cin, cout = 16, 8
        module = EMCAD_EUCB(in_channels=cin, out_channels=cout)
        inputs = (torch.rand(1, cin, 32, 32),)  # x

    elif name == "sh":
        cin = 16
        module = EMCAD_SegHead(in_channels=cin, num_classes=2)
        inputs = (torch.rand(1, cin, 32, 32),)  # x

    else:
        raise ValueError(f"Unknown module name: {name}")

    return module, inputs


if __name__ == "__main__":
    # 这里切换要测试的模块名字:
    # 可选: "mscb", "mscam", "lgag", "eucb", "sh"
    module_name = "eucb"

    model, inputs = build_test_module(module_name)

    print(f"🔧 Testing EMCAD module: {module_name}")

    # --- Forward 测试 ---
    try:
        with torch.no_grad():
            out = model(*inputs)
        in_shapes = ", ".join([str(t.shape) for t in inputs])
        print(f"✅ Forward Pass Success: {in_shapes} → {tuple(out.shape)}")
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
