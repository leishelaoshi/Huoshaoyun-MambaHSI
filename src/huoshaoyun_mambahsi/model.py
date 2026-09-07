from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class StableGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        n, c = x.shape[:2]
        y = x.reshape(n, self.num_groups, c // self.num_groups, -1)
        base_dtype = y.dtype
        z = y.float() if base_dtype in (torch.float16, torch.bfloat16) else y
        mean = z.mean(dim=(2, 3), keepdim=True)
        var = (z - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        z = (z - mean) * torch.rsqrt(var + self.eps)
        y = z.to(base_dtype).reshape_as(x)
        shape = [1, c] + [1] * (x.ndim - 2)
        return (y * self.weight.view(*shape) + self.bias.view(*shape)).contiguous()


class SafeAvgPool2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] < 2 or x.shape[-1] < 2:
            return x
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class SpeMamba(nn.Module):
    def __init__(
        self,
        channels: int,
        token_num: int = 4,
        group_num: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_channels = channels
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = token_num * self.group_channel_num
        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Sequential(
            StableGroupNorm(group_num, self.channel_num),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        if c < self.channel_num:
            x = torch.cat(
                [x, x.new_zeros((b, self.channel_num - c, h, w))], dim=1
            )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(b * h * w, self.token_num, self.group_channel_num)
        x = self.mamba(x.contiguous()).contiguous()
        x = x.reshape(b, h, w, self.channel_num).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)[:, : self.input_channels]
        return (residual + x).contiguous()


class SpaMamba(nn.Module):
    def __init__(
        self,
        channels: int,
        group_num: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Sequential(
            StableGroupNorm(group_num, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().reshape(b, h * w, c)
        y = self.mamba(y.contiguous()).contiguous()
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return (residual + self.proj(y)).contiguous()


class BothMamba(nn.Module):
    def __init__(
        self,
        channels: int,
        token_num: int = 4,
        group_num: int = 4,
        use_att: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.use_att = use_att
        self.spa = SpaMamba(channels, group_num, d_state, d_conv, expand)
        self.spe = SpeMamba(channels, token_num, group_num, d_state, d_conv, expand)
        if use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spa = self.spa(x)
        spe = self.spe(x)
        if self.use_att:
            w = torch.softmax(self.weights, dim=0)
            fused = w[0] * spa + w[1] * spe
        else:
            fused = spa + spe
        return (fused + x).contiguous()


class MambaHSI(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        mamba_type: str = "both",
        token_num: int = 4,
        group_num: int = 4,
        use_att: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            StableGroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        def block() -> nn.Module:
            if mamba_type == "spa":
                return SpaMamba(hidden_dim, group_num, d_state, d_conv, expand)
            if mamba_type == "spe":
                return SpeMamba(
                    hidden_dim, token_num, group_num, d_state, d_conv, expand
                )
            if mamba_type == "both":
                return BothMamba(
                    hidden_dim,
                    token_num,
                    group_num,
                    use_att,
                    d_state,
                    d_conv,
                    expand,
                )
            raise ValueError(f"Unsupported mamba_type: {mamba_type}")

        self.encoder = nn.Sequential(
            block(),
            SafeAvgPool2d(),
            block(),
            SafeAvgPool2d(),
            block(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=1),
            StableGroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x.contiguous())
        x = self.encoder(x)
        return self.head(x).contiguous()


class MambaClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mamba_type: str = "both",
        use_att: bool = True,
        num_classes: int = 2,
        hidden_dim: int = 64,
        token_num: int = 4,
        group_num: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MambaHSI(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            mamba_type=mamba_type,
            token_num=token_num,
            group_num=group_num,
            use_att=use_att,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if out.ndim != 4 or out.shape[1] != self.num_classes:
            raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")
        h, w = out.shape[-2:]
        return out[:, :, h // 2, w // 2].contiguous()
