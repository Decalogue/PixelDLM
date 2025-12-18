"""
像素级解码器模块
基于 DiP 和 DeCo 论文的实现

1. U-Net 解码器（DiP 方案）：Post-hoc Refinement
2. 可选的 Transformer 解码器（DeCo 方案）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UNetDecoder(nn.Module):
    """
    U-Net 像素级解码器（基于 DiP 论文）
    
    架构：
    - 编码器：下采样路径（提取特征）
    - 解码器：上采样路径（生成像素）
    - 条件注入：DiT 输出特征 concatenate 到最深层
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # 输入图像通道数
        out_channels: int = 3,  # 输出图像通道数
        base_channels: int = 64,  # 基础通道数
        cond_dim: int = 768,  # 条件特征维度（DiT embed_dim）
        depth: int = 4,  # U-Net 深度
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.depth = depth
        
        # 编码器（下采样）
        self.encoder = nn.ModuleList()
        self.encoder_pool = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            if i < depth - 1:
                self.encoder_pool.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        # 最深层：注入条件信息（DiT 输出）
        # 将条件特征从 [B, num_patches, embed_dim] 转换为空间特征图
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, base_channels * (2 ** (depth - 1))),
            nn.ReLU(inplace=True),
        )
        
        # 解码器（上采样）
        # 注意：解码器层的输入通道数需要考虑上一步上采样后的输出
        # 对于解码器层 i（从深到浅）：
        #   - 最深层（i=depth-1）：输入 = 编码器输出 + 条件 = base_channels * 2^(depth-1) * 2
        #   - 其他层（i<depth-1）：输入 = 上一步上采样后输出 + skip connection
        #     上一步上采样后输出 = base_channels * 2^i（因为上一步输出 base_channels * 2^(i+1)，上采样后通道数不变）
        #     skip connection = base_channels * 2^i
        #     所以 in_ch = base_channels * 2^i + base_channels * 2^i = base_channels * 2^(i+1)
        self.decoder = nn.ModuleList()
        self.decoder_up = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            if i == depth - 1:
                # 最深层：接收条件信息
                in_ch = base_channels * (2 ** i) * 2  # *2 因为 concatenate 条件
            else:
                # 其他层：上一步上采样后的输出 + skip connection
                # 上一步上采样后输出 = base_channels * 2^i（上采样不改变通道数）
                # skip connection = base_channels * 2^i
                in_ch = base_channels * (2 ** i) + base_channels * (2 ** i)  # = base_channels * 2^(i+1)
            
            out_ch = base_channels * (2 ** i) if i > 0 else out_channels
            
            if i > 0:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, base_channels * (2 ** i), 3, padding=1),
                        nn.BatchNorm2d(base_channels * (2 ** i)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_channels * (2 ** i), base_channels * (2 ** i), 3, padding=1),
                        nn.BatchNorm2d(base_channels * (2 ** i)),
                        nn.ReLU(inplace=True),
                    )
                )
                self.decoder_up.append(nn.ConvTranspose2d(
                    base_channels * (2 ** i),
                    base_channels * (2 ** (i - 1)),
                    2,
                    stride=2
                ))
            else:
                # 最后一层：输出
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, base_channels, 3, padding=1),
                        nn.BatchNorm2d(base_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_channels, out_channels, 3, padding=1),
                        nn.Tanh(),  # 输出范围 [-1, 1]，训练时需要调整
                    )
                )
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            condition: 条件特征 [B, num_patches, embed_dim]（DiT 输出）
        
        Returns:
            output: 输出图像 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 编码器路径（下采样）
        encoder_features = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            encoder_features.append(x)
            if i < len(self.encoder_pool):
                x = self.encoder_pool[i](x)
        
        # 最深层：注入条件信息
        if condition is not None:
            # 将条件特征转换为空间特征图
            # 假设 condition 是 [B, num_patches, embed_dim]
            B_cond, num_patches, embed_dim = condition.shape
            
            # 投影条件特征
            cond_proj = self.cond_proj(condition)  # [B, num_patches, base_channels * 2^(depth-1)]
            
            # 假设 num_patches 对应一个空间维度（例如 16x16 = 256）
            # 需要 reshape 到空间特征图
            grid_size = int(num_patches ** 0.5)
            if grid_size * grid_size == num_patches:
                # 完全平方数，可以 reshape
                cond_spatial = cond_proj.view(B_cond, grid_size, grid_size, -1)
                cond_spatial = cond_spatial.permute(0, 3, 1, 2)  # [B, C, H, W]
                # 如果尺寸不匹配，需要插值
                if cond_spatial.shape[2:] != x.shape[2:]:
                    cond_spatial = F.interpolate(
                        cond_spatial,
                        size=x.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
            else:
                # 不是完全平方数，使用平均池化然后插值
                cond_spatial = cond_proj.mean(dim=1, keepdim=True)  # [B, 1, C]
                cond_spatial = cond_spatial.unsqueeze(-1).unsqueeze(-1)  # [B, 1, C, 1, 1]
                cond_spatial = cond_spatial.expand(-1, -1, -1, x.shape[2], x.shape[3])
                cond_spatial = cond_spatial.squeeze(1)  # [B, C, H, W]
            
            # Concatenate 条件到最深层特征
            x = torch.cat([x, cond_spatial], dim=1)
        
        # 解码器路径（上采样）
        # 解码器列表是从深到浅构建的：depth-1, depth-2, ..., 0
        # 编码器列表是从浅到深构建的：0, 1, ..., depth-1
        for i, decoder_block in enumerate(self.decoder):
            # 解码器列表索引 i 对应解码器层 (depth-1-i)
            decoder_layer = self.depth - 1 - i  # 当前解码器层（从深到浅）
            
            # Skip connection: 在 decoder_block 之前 concatenate
            # 最深层（depth-1）已经和条件 concatenate，不需要 skip
            if decoder_layer < self.depth - 1:  # 不是最深层，需要 skip connection
                encoder_layer = decoder_layer  # 连接到相同深度的编码器层
                if encoder_layer >= 0 and encoder_layer < len(encoder_features):
                    # 调整 skip connection 的尺寸以匹配当前 x
                    skip_feature = encoder_features[encoder_layer]
                    if x.shape[2:] != skip_feature.shape[2:]:
                        skip_feature = F.interpolate(
                            skip_feature,
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    x = torch.cat([x, skip_feature], dim=1)
            
            # 执行 decoder_block
            x = decoder_block(x)
            
            # 上采样（如果不是最后一层）
            if i < len(self.decoder_up):
                x = self.decoder_up[i](x)
        
        return x


class PixelDecoderUNet(nn.Module):
    """
    U-Net 像素级解码器包装类
    用于在 DiT 输出后细化图像
    """
    
    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        cond_dim: int = 768,
        depth: int = 3,  # 对于 64x64 图像，depth=3 足够
    ):
        super().__init__()
        
        self.unet = UNetDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            cond_dim=cond_dim,
            depth=depth,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]，值范围 [0, 1] 或 [-1, 1]
            condition: 条件特征 [B, num_patches, embed_dim]（DiT 输出）
        
        Returns:
            output: 细化后的图像 [B, C, H, W]
        """
        # 如果输入是 [-1, 1]，转换为 [0, 1]
        if x.min() < 0:
            x = (x + 1) / 2
        
        # U-Net 处理
        output = self.unet(x, condition)
        
        # 如果输出是 [-1, 1]，转换为 [0, 1]
        if output.min() < 0:
            output = (output + 1) / 2
        
        # 限制到 [0, 1] 范围
        output = torch.clamp(output, 0, 1)
        
        return output
