import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Focal Modulation Module
#   Paper: "Focal Modulation Networks", NeurIPS 2022
#   Official repo: https://github.com/microsoft/FocalNet
#
#   这是基于论文公式的简化实现，保留了核心结构：
#   - 多层 depthwise conv 做分层语境化 (hierarchical context)
#   - gating 聚合多尺度 & 全局上下文
#   - 线性映射得到调制器，按元素乘到 query 上
#
#   输入 / 输出: (B, C, H, W) -> (B, C, H, W)
# ============================================================


class FocalModulation(nn.Module):
    """
    Focal Modulation module (simplified, PyTorch, NCHW).

    Args:
        dim: 通道数 C
        focal_levels: 多尺度层数 L（论文中一般是 2~4）
        kernel_sizes: 每一层 depthwise conv 的 kernel size 列表，长度要等于 focal_levels
        use_post_ln: 是否在输出后加一层 LayerNorm (channels_last)，方便插到 ViT 类结构
        act_layer: 激活函数
    """

    def __init__(
        self,
        dim: int,
        focal_levels: int = 3,
        kernel_sizes=None,
        use_post_ln: bool = False,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.focal_levels = focal_levels
        if kernel_sizes is None:
            # 默认从小到大几个 kernel
            kernel_sizes = [3, 5, 7][:focal_levels]
        assert len(kernel_sizes) == focal_levels, "kernel_sizes length must equal focal_levels"

        # 1) f_z: 输入投影到 Z^0
        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # 2) L 层 depthwise conv + 激活: Z^ell = GELU(DWConv(Z^{ell-1}))
        self.dw_convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.dw_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=k, padding=padding, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    act_layer(),
                )
            )

        # 3) gating: 从原始 X 产生 (L+1) 个 gating map
        #    形状: (B, L+1, H, W)，每个通道对应一个尺度的 gate
        self.gating = nn.Conv2d(dim, focal_levels + 1, kernel_size=1, bias=True)

        # 4) h: 把聚合后的 Z_out -> 调制器 M (同通道数)
        self.modulator_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # 5) q: query projection
        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # 可选: 输出后做一次 LayerNorm (channels_last)
        self.use_post_ln = use_post_ln
        if use_post_ln:
            self.ln = nn.LayerNorm(dim)

        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ---- 1) 投影得到 Z^0 ----
        z = self.proj_in(x)  # (B, C, H, W)

        # ---- 2) 分层语境化 Z^ell ----
        # 保存每一层的特征，最后一层再拿去做 global pooling
        zs = []
        cur = z
        for dw in self.dw_convs:
            cur = dw(cur)
            zs.append(cur)  # 每个都是 (B, C, H, W)

        # ---- 3) 全局上下文 Z^{L+1} ----
        # 对最后一层做 GAP，再 broadcast 回 H x W
        z_global = cur.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        z_global = z_global.expand(-1, -1, H, W)       # (B, C, H, W)

        # 一共 L+1 个尺度特征
        zs.append(z_global)  # len(zs) = L + 1

        # ---- 4) gating 聚合 ----
        gate_logits = self.gating(x)  # (B, L+1, H, W)
        # 可以用 sigmoid 保证在 0~1 范围，也可以 softmax，这里用 sigmoid 足够简单
        gates = torch.sigmoid(gate_logits)

        # 对每个尺度做加权求和
        # Z_out = sum_{ell}( G^ell * Z^ell )
        z_out = 0.0
        for level_idx in range(self.focal_levels + 1):
            g_l = gates[:, level_idx : level_idx + 1, :, :]  # (B, 1, H, W)
            z_l = zs[level_idx]                              # (B, C, H, W)
            z_out = z_out + g_l * z_l                        # broadcast on channels

        # ---- 5) 通道维度上的调制器 M ----
        m = self.modulator_proj(z_out)  # (B, C, H, W)
        m = self.act(m)

        # ---- 6) query projection + 元素级调制 ----
        q = self.query_proj(x)          # (B, C, H, W)
        y = q * m                       # Focal Modulation: q(x) ⊙ M(x, X)

        # 可选: 输出层做 LayerNorm (channels_last)
        if self.use_post_ln:
            y_perm = y.permute(0, 2, 3, 1)   # (B, H, W, C)
            y_perm = self.ln(y_perm)
            y = y_perm.permute(0, 3, 1, 2)   # 回到 (B, C, H, W)

        return y


# ============================================================
# 一个简单的 Block 封装: norm + FocalModulation + residual
#   方便你直接在 backbone / UNet 里替换原来的 self-attention block
# ============================================================

class FocalModulationBlock(nn.Module):
    """
    标准 Block: (Norm -> FocalModulation -> Dropout) + Residual

    输入 / 输出: (B, C, H, W)
    """

    def __init__(
        self,
        dim: int,
        focal_levels: int = 3,
        kernel_sizes=None,
        drop: float = 0.0,
        use_post_ln: bool = False,
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.focal = FocalModulation(
            dim=dim,
            focal_levels=focal_levels,
            kernel_sizes=kernel_sizes,
            use_post_ln=use_post_ln,
        )
        self.drop = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.focal(out)
        out = self.drop(out)
        out = out + residual
        return out


# ============================================================
# 测试脚本 (和你 EcNet 小脚本同风格)
#   - Forward shape 测试
#   - NNI 统计 FLOPs / Params (可选)
# ============================================================

if __name__ == "__main__":
    # 你可以在这里切换测试 FocalModulation 或 FocalModulationBlock
    TEST_BLOCK = "focal_mod"  # "focal_mod" 或 "block"

    b = 1
    input_size = 16
    dim = 16

    x = torch.rand(b, dim, input_size, input_size)  # 输入: (B, C, H, W)

    if TEST_BLOCK == "focal_mod":
        model = FocalModulation(dim=dim, focal_levels=3)
    else:
        model = FocalModulationBlock(dim=dim, focal_levels=3, drop=0.0)

    print(f"🔧 Testing: {TEST_BLOCK}")
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
        print(f"📊 FLOPs:  {flops / 1e6:.2f} MFLOPs | Params: {params / 1e6:.4f} M")
    except ImportError:
        print("⚠️ NNI not installed. Run: pip install nni")
    except Exception as e:
        print(f"⚠️ FLOPs/Params counting failed: {e}")
