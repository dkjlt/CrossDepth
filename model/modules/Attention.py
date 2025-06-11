import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        x = self.proj(x)  # (B, C, H', W')
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (H, W)  # (B, N, C), (H, W)

class PatchUnembed(nn.Module):
    def __init__(self, out_channels, patch_size):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(self, x, shape):
        B, N, C = x.shape
        H, W = shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        return x

class TransformerFusion(nn.Module):
    def __init__(self, patch_size=16, embed_dim=64, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Project Ddff + Udff and Drgb to patch tokens
        self.embed_dff = PatchEmbed(2, embed_dim, patch_size)
        self.embed_rgb = PatchEmbed(1, embed_dim, patch_size)

        # Self-Attention: refine Ddff+Udff
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True
        )

        # Cross-Attention: Drgb queries Ddff features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Output correction
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)  # predict per-patch residual
        )

        self.unembed = PatchUnembed(1, patch_size)

    def forward(self, D_rgb, D_dff, U_dff):
        # 1. 将 Ddff 与其不确定性合并
        x_dff = torch.cat([D_dff, U_dff], dim=1)  # (B, 2, H, W)
        import pdb; pdb.set_trace()

        B, _, H, W = D_rgb.shape
        dff_tokens, shape = self.embed_dff(x_dff)    # (B, N, C)
        rgb_tokens, _ = self.embed_rgb(D_rgb)        # (B, N, C)

        # 2. Self-Attention over DFF tokens
        dff_refined = self.self_attn(dff_tokens)     # (B, N, C)

        # 3. Cross-Attention from DFF ← RGB (反过来)
        dff_fused, _ = self.cross_attn(dff_refined, rgb_tokens, rgb_tokens)

        # 4. Predict per-patch correction → 应用于 Ddff
        delta = self.head(dff_fused)  # (B, N, 1)

        delta_img = self.unembed(delta, shape)  # (B, 1, H, W)

        D_fused = D_dff + delta_img

        D_fused = torch.clamp(D_fused, 0.0, 1.0)

        return D_fused, delta_img
