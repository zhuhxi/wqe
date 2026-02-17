import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EDiT: Efficient Diffusion Transformer style modules (2025)
#
# This file provides a practical, plug-and-play implementation:
#   - EDiTBlock: one DiT-style block for latent feature maps.
#   - EDiT: a small backbone made of stacked EDiTBlock.
#
# Input / output for both classes:
#   (B, C, H, W) -> (B, C, H, W)
# ============================================================


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN modulation:
      x_mod = x * (1 + scale) + shift
    x:     (B, N, C)
    shift: (B, C)
    scale: (B, C)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class EDiTBlock(nn.Module):
    """
    DiT-style block with:
      - local depthwise conv branch (efficient local prior)
      - self-attention
      - optional cross-attention
      - MLP
      - AdaLN-style conditioning from `cond`

    Args:
        dim: feature channels C
        num_heads: number of attention heads
        mlp_ratio: expansion ratio for MLP hidden dim
        cond_dim: conditioning vector dim; if None, use dim
        with_cross_attn: whether to enable cross-attention
        dropout: dropout for attention/MLP
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: int = None,
        with_cross_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.with_cross_attn = with_cross_attn

        # Local branch to keep convolutional inductive bias.
        self.local_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.local_scale = nn.Parameter(torch.zeros(1))

        # Token mixing.
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if with_cross_attn:
            self.norm_ca = nn.LayerNorm(dim, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # AdaLN-style modulation:
        # self-attn: shift/scale/gate
        # cross-attn: shift/scale/gate (optional)
        # mlp: shift/scale/gate
        cond_dim = dim if cond_dim is None else cond_dim
        num_chunks = 9 if with_cross_attn else 6
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_chunks * dim),
        )

        # AdaLN-zero style init: start close to identity for conditioned path.
        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)

    def _build_modulation(self, cond: torch.Tensor, batch_size: int, device, dtype):
        zeros = torch.zeros(batch_size, self.dim, device=device, dtype=dtype)
        ones = torch.ones(batch_size, self.dim, device=device, dtype=dtype)

        if cond is None:
            if self.with_cross_attn:
                return zeros, zeros, ones, zeros, zeros, ones, zeros, zeros, ones
            return zeros, zeros, ones, zeros, zeros, ones

        mod = self.cond_proj(cond)
        return mod.chunk(9 if self.with_cross_attn else 6, dim=-1)

    def _ensure_tokens(self, context: torch.Tensor) -> torch.Tensor:
        if context is None:
            return None
        if context.dim() == 3:
            # (B, T, C)
            return context
        if context.dim() == 4:
            # (B, C, H, W) -> (B, T, C)
            return context.flatten(2).transpose(1, 2)
        raise ValueError("context must be (B, T, C) or (B, C, H, W)")

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: (B, C, H, W)
        cond: (B, cond_dim), optional
        context: optional cross-attn context, (B, T, C) or (B, C, H, W)
        """
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"Expected dim={self.dim}, but got channels={c}")

        # Local residual branch.
        x = x + self.local_scale * self.local_dw(x)

        # Spatial map -> token sequence.
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C), N=H*W

        mods = self._build_modulation(cond, b, x.device, x.dtype)
        if self.with_cross_attn:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_ca,
                scale_ca,
                gate_ca,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = mods
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods

        # Self-attention.
        h_msa = _modulate(self.norm1(tokens), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(h_msa, h_msa, h_msa, need_weights=False)
        tokens = tokens + gate_msa.unsqueeze(1) * attn_out

        # Optional cross-attention.
        if self.with_cross_attn:
            ctx = self._ensure_tokens(context)
            if ctx is not None:
                h_ca = _modulate(self.norm_ca(tokens), shift_ca, scale_ca)
                ca_out, _ = self.cross_attn(h_ca, ctx, ctx, need_weights=False)
                tokens = tokens + gate_ca.unsqueeze(1) * ca_out

        # MLP.
        h_mlp = _modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(h_mlp)

        # Token sequence -> spatial map.
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return out


class EDiT(nn.Module):
    """
    Small EDiT backbone: stack of EDiTBlock with residual output head.

    Args:
        dim: feature channels C
        depth: number of EDiT blocks
        num_heads: attention heads in each block
        mlp_ratio: MLP expansion ratio
        cond_dim: conditioning vector dim; if None, use dim
        with_cross_attn: whether blocks use cross-attention
    """

    def __init__(
        self,
        dim: int = 320,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_dim: int = None,
        with_cross_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            [
                EDiTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cond_dim=cond_dim,
                    with_cross_attn=with_cross_attn,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.GroupNorm(
            num_groups=self._auto_group_count(dim),
            num_channels=dim,
            eps=1e-6,
        )
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    @staticmethod
    def _auto_group_count(channels: int, max_groups: int = 32) -> int:
        g = min(max_groups, channels)
        while g > 1 and channels % g != 0:
            g -= 1
        return g

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.size(1) != self.dim:
            raise ValueError(f"Expected dim={self.dim}, but got channels={x.size(1)}")

        residual = x
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, cond=cond, context=context)
        x = self.out_proj(F.silu(self.out_norm(x)))
        return residual + x


def build_test_module(name: str):
    """
    Supported names:
      - "edit_block": EDiTBlock
      - "edit": EDiT
    """
    name = name.lower()
    b, c, h, w = 1, 64, 16, 16
    x = torch.rand(b, c, h, w)
    cond = torch.rand(b, c)
    context = torch.rand(b, 77, c)  # text-like context tokens

    if name == "edit_block":
        module = EDiTBlock(
            dim=c,
            num_heads=8,
            mlp_ratio=4.0,
            cond_dim=c,
            with_cross_attn=True,
        )
        inputs = (x, cond, context)
    elif name == "edit":
        module = EDiT(
            dim=c,
            depth=3,
            num_heads=8,
            mlp_ratio=4.0,
            cond_dim=c,
            with_cross_attn=True,
        )
        inputs = (x, cond, context)
    else:
        raise ValueError(f"Unknown module name: {name}")

    return module, inputs


if __name__ == "__main__":
    # Switch between:
    # "edit_block", "edit"
    module_name = "edit"
    model, inputs = build_test_module(module_name)
    in_shapes = ", ".join(str(t.shape) for t in inputs)

    print(f"Testing EDiT module: {module_name}")
    print(f"Input shape(s): {in_shapes}")

    # Forward shape check.
    try:
        with torch.no_grad():
            out = model(*inputs)
        print(f"Forward Pass Success: {in_shapes} -> {tuple(out.shape)}")
    except Exception as e:
        print(f"Forward Failed: {e}")

    # FLOPs / Params (optional, requires nni).
    try:
        from nni.compression.utils.counter import count_flops_params

        flops, params, _ = count_flops_params(model, x=inputs)
        print(f"FLOPs: {flops / 1e6:.2f} MFLOPs | Params: {params / 1e6:.4f} M")
    except ImportError:
        print("NNI not installed. Run: pip install nni")
    except Exception as e:
        print(f"FLOPs/Params counting failed: {e}")
