import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Global Response Normalization (ConvNeXt V2)
#    Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
#    X_out = X + gamma * (X * N(G(X))) + beta
#    G(X) : L2 norm over spatial dims, then normalized across channels
# ============================================================

class GRN(nn.Module):
    """
    Global Response Normalization (GRN) layer.
    适用输入: (B, C, H, W)，不改变通道数和分辨率。

    dim: 通道数 C
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 1) 空间维度上的 L2 范数  G(x) ∈ (B, C, 1, 1)
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        # 2) 在通道维度上做归一化，形成通道间竞争
        #    N(Gx)_c = Gx_c / (mean_c' Gx_c' + eps)
        Gx_mean = Gx.mean(dim=1, keepdim=True)
        Nx = Gx / (Gx_mean + self.eps)

        # 3) 校准原始响应并加 residual
        return x + self.gamma * (x * Nx) + self.beta


# ============================================================
# 2. Multi-scale Depthwise Conv Block (参考 MRLPFNet & EMCAD)
#    - 多个不同 kernel size 的 depthwise conv
#    - 聚合后做一次 pointwise conv
#    - 保持输入输出 shape 一致: (B, C, H, W) -> (B, C, H, W)
# ============================================================

class MultiScaleDWConvBlock(nn.Module):
    """
    Multi-scale Depthwise Convolution Block.

    参考:
      - "Multi-scale Residual Low-Pass Filter Network for Image Deblurring" (ICCV 2023)
      - "EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation" (CVPR 2024)
    但实现是简化版，方便你当通用小模块插在 UNet / EcNet 里。

    Args:
        dim: 输入/输出通道数 C
        kernel_sizes: 多尺度 depthwise 卷积的 kernel size 列表
        use_grn: 是否在 block 内部叠一层 GRN
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes=(3, 5, 7),
        use_grn: bool = True,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim

        # depthwise conv 分支
        dw_convs = []
        for k in kernel_sizes:
            padding = k // 2
            dw_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=k, padding=padding, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    act_layer(),
                )
            )
        self.dw_convs = nn.ModuleList(dw_convs)

        # 聚合后的 pointwise conv
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(dim)

        self.act = act_layer()
        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        residual = x

        # 多尺度 depthwise 卷积，结果求和
        out = 0
        for branch in self.dw_convs:
            out = out + branch(x)
        out = out / len(self.dw_convs)

        # pointwise 融合
        out = self.pw_bn(self.pw(out))

        # 可选 GRN
        if self.use_grn:
            out = self.grn(out)

        # 残差 + 激活
        out = out + residual
        out = self.act(out)
        return out


# ============================================================
# 3. 通用测试脚本
#    - 跟你发的 EcNet 小脚本同一个风格
#    - Forward shape 检查
#    - NNI 统计 FLOPs / Params
# ============================================================

def build_test_module(name: str):
    """
    根据名字构造一个待测试的“小模块”以及对应的输入 shape。
    你后面抠新模块，就在这里加一个 elif 分支即可。
    """
    if name.lower() == "grn":
        dim = 16
        module = GRN(dim=dim)
        input_shape = (1, dim, 16, 16)

    elif name.lower() == "msdw":
        dim = 16
        module = MultiScaleDWConvBlock(dim=dim)
        input_shape = (1, dim, 16, 16)

    else:
        raise ValueError(f"Unknown module name: {name}")

    return module, input_shape


if __name__ == "__main__":
    # 你可以在这里切换要测试的模块名字: "grn" 或 "msdw"
    module_name = "msdw"  # 改成 "grn" 就能测 GRN

    # ---- 构造模块 & 随机输入 ----
    model, shape = build_test_module(module_name)
    b, c, h, w = shape
    x = torch.rand(b, c, h, w)

    print(f"🔧 Testing module: {module_name}")
    print(f"   Input  shape: {tuple(x.shape)}")

    # --- Shape 测试 ---
    try:
        out = model(x)
        print(f"✅ Forward Pass Success: {x.shape} → {out.shape}")
    except Exception as e:
        print(f"❌ Forward Failed: {e}")

    # --- FLOPs 和 参数统计 ---
    try:
        from nni.compression.utils.counter import count_flops_params

        flops, params, _ = count_flops_params(model, x=(x,))
        print(f"📊 FLOPs:  {flops / 1e6:.2f} MFLOPs")
        print(f"📦 Params: {params / 1e6:.2f} M")
    except ImportError:
        print("⚠️ NNI not installed. Run: pip install nni")
    except Exception as e:
        print(f"⚠️ FLOPs/Params counting failed: {e}")
