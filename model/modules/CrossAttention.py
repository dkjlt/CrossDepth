import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H//p, W//p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        out = query + self.attn(query, key_value, key_value)[0]
        out = self.norm(out)
        return out


class CrossViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.self_d = TransformerEncoder(embed_dim, num_heads)
        self.self_rgb = TransformerEncoder(embed_dim, num_heads)
        self.cross_d = CrossAttention(embed_dim, num_heads)
        self.cross_rgb = CrossAttention(embed_dim, num_heads)

    def forward(self, d_feat, rgb_feat):
        d_feat = self.self_d(d_feat)
        rgb_feat = self.self_rgb(rgb_feat)

        d_feat = self.cross_d(d_feat, rgb_feat)
        rgb_feat = self.cross_rgb(rgb_feat, d_feat)

        return d_feat, rgb_feat


class DepthFusionCrossViT(nn.Module):
    def __init__(self, in_channels_d=2, in_channels_rgb=2, embed_dim=32, patch_size=4, depth=2):
        super().__init__()
        self.embed_d = PatchEmbed(in_channels_d, embed_dim, patch_size)
        self.embed_rgb = PatchEmbed(in_channels_rgb, embed_dim, patch_size)

        self.blocks = nn.ModuleList([
            CrossViTBlock(embed_dim) for _ in range(depth)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * 2, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, ddff, drgb, udff):
        d_input = torch.cat([ddff, udff], dim=1)
        rgb_input = torch.cat([drgb, udff], dim=1)

        B, _, H, W = ddff.shape

        d_tokens = self.embed_d(d_input)     # (B, N, C)
        rgb_tokens = self.embed_rgb(rgb_input)

        for block in self.blocks:
            d_tokens, rgb_tokens = block(d_tokens, rgb_tokens)

        fused = torch.cat([d_tokens, rgb_tokens], dim=-1)  # (B, N, 2C)

        # reshape to 2D feature map
        num_patches = fused.shape[1]
        h_p = H // 4
        w_p = W // 4
        fused_map = fused.transpose(1, 2).reshape(B, -1, h_p, w_p)

        out = self.decoder(fused_map)
        out = torch.clamp(out, 0.02, 0.28)

        return out
