import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 导入像素级解码器
try:
    from pixel_decoder import PixelDecoderUNet
    HAS_PIXEL_DECODER = True
except ImportError:
    HAS_PIXEL_DECODER = False


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 数值稳定性：确保 norm 不会太小
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        norm = torch.clamp(norm, min=self.eps)  # 确保 norm >= eps
        return x / norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Activation"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with Zero Initialization"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = RMSNorm(embed_dim, eps=1e-6)
        self.scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.shift_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True)
        )
        
        # Zero initialization
        nn.init.zeros_(self.scale_mlp[-1].weight)
        nn.init.zeros_(self.scale_mlp[-1].bias)
        nn.init.zeros_(self.shift_mlp[-1].weight)
        nn.init.zeros_(self.shift_mlp[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift = self.shift_mlp(c)
        scale = self.scale_mlp(c) + 1.0
        # 数值稳定性：限制 scale 的范围，防止过大或过小
        scale = torch.clamp(scale, min=0.1, max=10.0)
        return self.norm(x) * scale.unsqueeze(1) + shift.unsqueeze(1)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional attention mask"""
    def __init__(self, embed_dim: int, num_heads: int = 16, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C]
            attention_mask: Attention mask [B, N] - 1 表示有效，0 表示 padding
        """
        B, N, C = x.shape
        
        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [B, N] -> [B, 1, 1, N]
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            # Mask out padding positions: set to large negative value before softmax
            attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
            # 数值稳定性：检查是否所有位置都是 -inf（全 padding 的情况）
            # 如果所有位置都是 -inf，softmax 会产生 nan，需要特殊处理
            all_inf = (attn == float('-inf')).all(dim=-1, keepdim=True)
            if all_inf.any():
                # 对于全 padding 的情况，设置为均匀分布
                attn = attn.masked_fill(all_inf, 0.0)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with AdaLN-Zero"""
    def __init__(self, embed_dim: int, num_heads: int = 16, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0):
        super().__init__()
        
        mlp_dim = int(embed_dim * mlp_ratio)
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.adaLN = AdaLNZero(embed_dim)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with AdaLN-Zero
        x = x + self.attn(self.adaLN(x, c), attention_mask=attention_mask)
        
        # MLP with AdaLN-Zero
        x = x + self.mlp(self.adaLN(x, c))
        
        return x


class TimeEmbedding(nn.Module):
    """Time Step Embedding"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Sinusoidal embedding
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] time steps
        """
        emb = t.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb


class JiT(nn.Module):
    """
    Just image Transformer for Pixel-space Diffusion
    
    Key features:
    - Direct pixel space operation (no VAE)
    - Predicts clean data instead of noise
    - Large patch size for linear scaling
    - Transformer architecture
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        proj_dropout: float = 0.0,
        predict_clean: bool = True,
        max_cond_patches: int = 256,  # 条件图像的最大 patches 数
        use_pixel_decoder: bool = False,  # 是否使用像素级解码器（U-Net）
        pixel_decoder_depth: int = 3,  # U-Net 深度
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.predict_clean = predict_clean
        self.use_pixel_decoder = use_pixel_decoder and HAS_PIXEL_DECODER
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.patch_grid_size = img_size // patch_size  # 例如 16 (16×16 patches)
        
        # 统一架构：固定最大序列长度
        # 总长度 = 条件最大长度 + 目标长度 = 256 + 256 = 512
        self.cond_max_patches = max_cond_patches
        self.max_seq_len = self.cond_max_patches + self.num_patches  # 例如 256 + 256 = 512
        
        # 存储 cond_max_patches 供后续使用
        self._cond_max_patches = max_cond_patches
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim, bias=False)
        
        # 2D Position embedding (简化的 2D 位置编码)
        # 分别对行和列进行编码，然后相加
        self.pos_embed_h = nn.Parameter(torch.zeros(1, self.patch_grid_size, embed_dim))  # 行位置编码
        self.pos_embed_w = nn.Parameter(torch.zeros(1, self.patch_grid_size, embed_dim))  # 列位置编码
        
        # Condition 2D position embedding (for conditional generation)
        max_cond_grid_size = int(math.sqrt(max_cond_patches))
        self.cond_pos_embed_h = nn.Parameter(torch.zeros(1, max_cond_grid_size, embed_dim))  # 条件行位置编码
        self.cond_pos_embed_w = nn.Parameter(torch.zeros(1, max_cond_grid_size, embed_dim))  # 条件列位置编码
        self.max_cond_grid_size = max_cond_grid_size
        
        # Time embedding
        self.time_embed = TimeEmbedding(embed_dim)
        
        # Condition embedding (for question image)
        self.condition_embed = nn.Linear(self.patch_dim, embed_dim, bias=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        
        # 像素级解码器（U-Net，基于 DiP）
        if self.use_pixel_decoder:
            self.pixel_decoder = PixelDecoderUNet(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=in_channels,
                base_channels=64,
                cond_dim=embed_dim,
                depth=pixel_decoder_depth,
            )
        else:
            self.pixel_decoder = None
        
        # Initialize
        nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.normal_(self.pos_embed_w, std=0.02)
        nn.init.normal_(self.cond_pos_embed_h, std=0.02)
        nn.init.normal_(self.cond_pos_embed_w, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def image_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches
        x: [B, C, H, W]
        returns: [B, num_patches, patch_dim]
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Image size must be {self.img_size}"
        
        # Reshape to patches
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H//p, W//p, C, p, p]
        x = x.reshape(B, self.num_patches, self.patch_dim)
        
        return x
    
    def patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to image
        patches: [B, num_patches, patch_dim]
        returns: [B, C, H, W]
        """
        B = patches.shape[0]
        p = self.patch_size
        n = int(math.sqrt(self.num_patches))
        
        # Reshape
        patches = patches.reshape(B, n, n, self.in_channels, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # [B, C, n, p, n, p]
        x = patches.reshape(B, self.in_channels, n * p, n * p)
        
        return x
    
    def _get_2d_pos_embed(self, num_patches: int, pos_embed_h: nn.Parameter, pos_embed_w: nn.Parameter) -> torch.Tensor:
        """
        计算 2D 位置编码
        
        Args:
            num_patches: patch 数量
            pos_embed_h: 行位置编码 [1, H, embed_dim]
            pos_embed_w: 列位置编码 [1, W, embed_dim]
        
        Returns:
            2D 位置编码 [1, num_patches, embed_dim]
        """
        # 假设是正方形 grid
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            # 如果不是完全平方数，使用近似
            grid_size = int(math.ceil(math.sqrt(num_patches)))
        
        H_p = min(grid_size, pos_embed_h.shape[1])
        W_p = min(grid_size, pos_embed_w.shape[1])
        
        pos_h = pos_embed_h[:, :H_p, :]  # [1, H_p, embed_dim]
        pos_w = pos_embed_w[:, :W_p, :]  # [1, W_p, embed_dim]
        
        # 广播相加：[1, H_p, 1, D] + [1, 1, W_p, D] = [1, H_p, W_p, D]
        pos_2d = pos_h.unsqueeze(2) + pos_w.unsqueeze(1)  # [1, H_p, W_p, embed_dim]
        pos_2d = pos_2d.reshape(1, H_p * W_p, self.embed_dim)  # [1, H_p*W_p, embed_dim]
        
        # 如果实际 num_patches 与计算的不同，需要调整
        if num_patches != pos_2d.shape[1]:
            if num_patches < pos_2d.shape[1]:
                pos_2d = pos_2d[:, :num_patches, :]
            else:
                # 填充（使用最后一个位置编码）
                padding = num_patches - pos_2d.shape[1]
                pos_padding = pos_2d[:, -1:, :].repeat(1, padding, 1)
                pos_2d = torch.cat([pos_2d, pos_padding], dim=1)
        
        return pos_2d
    
    def image_mask_to_patch_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        将图像 mask 转换为 patch mask
        
        Args:
            mask: Image mask [B, H, W] - 1 表示有效像素，0 表示 padding
        
        Returns:
            patch_mask: Patch mask [B, num_patches] - 1 表示有效 patch，0 表示 padding patch
        """
        B, H, W = mask.shape
        assert H == W == self.img_size, f"Mask size must be {self.img_size}"
        
        p = self.patch_size
        n = H // p
        
        # 将 mask 重塑为 patches，然后对每个 patch 取平均值
        # 如果 patch 中大部分像素是有效的，则认为 patch 是有效的
        mask_patches = mask.reshape(B, n, p, n, p)  # [B, n, p, n, p]
        mask_patches = mask_patches.mean(dim=(2, 4))  # [B, n, n] - 每个 patch 的平均值
        patch_mask = (mask_patches > 0.5).float()  # 阈值 0.5，大于 0.5 认为是有效 patch
        patch_mask = patch_mask.reshape(B, self.num_patches)  # [B, num_patches]
        
        return patch_mask
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Noisy image [B, C, H, W] or patches [B, num_patches, patch_dim]
            t: Time steps [B]
            condition: Condition image patches [B, cond_patches, patch_dim] (optional)
            mask: Image mask [B, H, W] - 1 表示有效像素，0 表示 padding (optional)
        
        Returns:
            - 如果 use_pixel_decoder=False: Predicted clean image patches [B, num_patches, patch_dim]
            - 如果 use_pixel_decoder=True: Refined image [B, C, H, W]
        """
        # Convert to patches if needed
        if x.dim() == 4:
            x_patches = self.image_to_patches(x)
        else:
            x_patches = x
        
        B = x_patches.shape[0]
        device = x_patches.device
        
        # Patch embedding
        x_target = self.patch_embed(x_patches)  # [B, num_patches, embed_dim]
        
        # 创建统一的 attention mask（固定长度 max_seq_len）
        attention_mask = None
        if condition is not None:
            # 有条件：前 cond_patches 个位置有效，剩余 padding 无效
            cond_patches_actual = min(condition.shape[1], self.cond_max_patches)
            cond_mask = torch.ones(B, cond_patches_actual, device=device)
            if cond_patches_actual < self.cond_max_patches:
                cond_padding_mask = torch.zeros(B, self.cond_max_patches - cond_patches_actual, device=device)
                cond_mask = torch.cat([cond_mask, cond_padding_mask], dim=1)
            # 目标 mask（如果有 mask，使用；否则全 1）
            if mask is not None:
                target_mask = self.image_mask_to_patch_mask(mask)  # [B, num_patches]
            else:
                target_mask = torch.ones(B, self.num_patches, device=device)
            attention_mask = torch.cat([cond_mask, target_mask], dim=1)  # [B, max_seq_len]
        else:
            # 无条件：前 cond_max_patches 个位置无效（padding），后 num_patches 个有效
            cond_mask = torch.zeros(B, self.cond_max_patches, device=device)
            if mask is not None:
                target_mask = self.image_mask_to_patch_mask(mask)  # [B, num_patches]
            else:
                target_mask = torch.ones(B, self.num_patches, device=device)
            attention_mask = torch.cat([cond_mask, target_mask], dim=1)  # [B, max_seq_len]
        
        # 2D Position embedding for target (简化的 2D 位置编码)
        target_pos_2d = self._get_2d_pos_embed(self.num_patches, self.pos_embed_h, self.pos_embed_w)
        x_target = x_target + target_pos_2d  # [B, num_patches, embed_dim]
        
        # Time embedding
        t_embed = self.time_embed(t)  # [B, embed_dim]
        
        # Condition embedding (统一架构：固定序列长度)
        if condition is not None:
            # 有条件：使用真实条件
            cond_embed = self.condition_embed(condition)  # [B, cond_patches, embed_dim]
            cond_patches = cond_embed.shape[1]
            
            # 为条件添加 2D 位置编码
            cond_pos_2d = self._get_2d_pos_embed(cond_patches, self.cond_pos_embed_h, self.cond_pos_embed_w)
            cond_embed = cond_embed + cond_pos_2d
            
            # Padding 到固定长度（cond_max_patches）
            if cond_patches < self.cond_max_patches:
                padding = self.cond_max_patches - cond_patches
                cond_padding = torch.zeros(B, padding, self.embed_dim, device=cond_embed.device, dtype=cond_embed.dtype)
                cond_embed = torch.cat([cond_embed, cond_padding], dim=1)  # [B, cond_max_patches, embed_dim]
            elif cond_patches > self.cond_max_patches:
                # 如果超过，截断
                cond_embed = cond_embed[:, :self.cond_max_patches, :]
        else:
            # 无条件：使用零 padding（统一架构）
            cond_embed = torch.zeros(B, self.cond_max_patches, self.embed_dim, device=x_target.device, dtype=x_target.dtype)
        
        # 拼接：总长度固定为 max_seq_len
        x = torch.cat([cond_embed, x_target], dim=1)  # [B, max_seq_len, embed_dim] = [B, 512, embed_dim]
        
        # Transformer blocks (使用统一的 attention mask)
        for block in self.blocks:
            x = block(x, t_embed, attention_mask=attention_mask)
        
        # Extract target patches (后 num_patches 个位置，固定)
        x_target_features = x[:, self.cond_max_patches:, :]  # [B, num_patches, embed_dim]
        
        # Output projection
        x = self.output_proj(x_target_features)
        x = self.proj_drop(x)
        
        # 如果提供了 mask，将 padding patches 的输出置零
        # 注意：这里 mask 只针对目标部分（已经提取出来了）
        if mask is not None:
            target_mask = self.image_mask_to_patch_mask(mask)  # [B, num_patches]
            target_mask_expanded = target_mask.unsqueeze(-1)  # [B, num_patches, 1]
            x = x * target_mask_expanded
        
        # 如果使用像素级解码器，进行细化（Post-hoc Refinement，DiP 方案）
        if self.use_pixel_decoder and self.pixel_decoder is not None:
            # 转换为图像格式 [B, C, H, W]
            img = self.patches_to_image(x)
            
            # 获取 DiT 的中间特征作为条件（使用 Transformer 输出的特征，而不是 output_proj 后的）
            # x_target_features: [B, num_patches, embed_dim] - 这是 Transformer blocks 的输出
            condition_features = x_target_features  # [B, num_patches, embed_dim]
            
            # U-Net 细化
            img = self.pixel_decoder(img, condition=condition_features)
            
            return img
        
        # 不使用像素解码器时，返回 patch 格式（保持向后兼容）
        return x
    
    def generate(
        self,
        condition: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate image using diffusion
        
        Args:
            condition: Condition image patches [B, cond_patches, patch_dim] (optional)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale (not used yet)
            device: Device to run on
        
        Returns:
            Generated image [B, C, H, W]
        """
        self.eval()
        
        B = 1 if condition is None else condition.shape[0]
        
        # 初始化目标部分（噪声）
        x_target = torch.randn(B, self.num_patches, self.patch_dim, device=device)
        
        # DDIM sampling
        with torch.no_grad():
            for i, step in enumerate(range(num_inference_steps)):
                t = torch.full((B,), step, dtype=torch.long, device=device)
                
                # Predict clean data (统一架构：forward 内部处理条件)
                x_0_pred = self.forward(x_target, t, condition=condition)
                
                # 判断输出格式
                if x_0_pred.dim() == 4:
                    # 图像格式（使用像素解码器）
                    x_0_pred_img = x_0_pred
                    # 需要转换回 patches 用于 DDIM 更新
                    x_0_pred_target = self.image_to_patches(x_0_pred_img)
                else:
                    # Patch 格式（不使用像素解码器）
                    x_0_pred_target = x_0_pred
                
                # DDIM update
                alpha = 1.0 - (step / num_inference_steps)
                if i < num_inference_steps - 1:
                    x_target = alpha * x_0_pred_target + (1 - alpha) * torch.randn_like(x_0_pred_target)
                else:
                    x_target = x_0_pred_target
        
        # Convert to image
        if self.use_pixel_decoder:
            # 如果使用像素解码器，最后一次 forward 已经返回图像
            img = self.forward(x_target, torch.zeros(B, dtype=torch.long, device=device), condition=condition)
            if img.dim() == 4:
                return img
        img = self.patches_to_image(x_target)
        
        return img


def build_jit_model(
    model_name: str = 'JiT-B/4',
    img_size: int = 64,
    predict_clean: bool = True,
    use_pixel_decoder: bool = False,
    pixel_decoder_depth: int = 3,
) -> JiT:
    """
    Build JiT model from config
    
    Model variants:
    - JiT-B/4: Base, patch_size=4 (适合 64×64 图像)
    - JiT-B/16: Base, patch_size=16 (适合 256×256 图像)
    - JiT-B/32: Base, patch_size=32
    - JiT-L/16: Large, patch_size=16
    - JiT-L/32: Large, patch_size=32
    - JiT-H/16: Huge, patch_size=16
    - JiT-H/32: Huge, patch_size=32
    """
    configs = {
        'JiT-B/4': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 4},
        'JiT-B/16': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 16},
        'JiT-B/32': {'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'patch_size': 32},
        'JiT-L/16': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 16},
        'JiT-L/32': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'patch_size': 32},
        'JiT-H/16': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'patch_size': 16},
        'JiT-H/32': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'patch_size': 32},
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = configs[model_name]
    
    model = JiT(
        img_size=img_size,
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        predict_clean=predict_clean,
        max_cond_patches=256,  # 默认支持最大 256 个条件 patches（64*64 图像，patch_size=4）
        use_pixel_decoder=use_pixel_decoder,
        pixel_decoder_depth=pixel_decoder_depth,
    )
    
    return model
