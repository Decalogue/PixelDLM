"""
优化的训练脚本
支持混合精度训练、梯度累积等优化
"""

import os
import argparse
import time
import math
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from model import build_jit_model
from dataset import TokenImageDataset


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def add_noise_to_timestep(x: torch.Tensor, t: torch.Tensor, noise_schedule: str = 'linear') -> Tuple[torch.Tensor, torch.Tensor]:
    """Add noise to image according to timestep"""
    B = x.shape[0]
    device = x.device
    
    noise = torch.randn_like(x)
    
    if noise_schedule == 'linear':
        alpha = 1.0 - (t.float() / 1000.0)
    else:
        alpha = 0.5 * (1 + torch.cos(torch.pi * t.float() / 1000.0))
    
    alpha = alpha.view(B, 1, 1, 1)
    noisy = alpha * x + (1 - alpha) * noise
    
    return noisy, noise


def train_epoch_optimized(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    rank: int = 0,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_wandb: bool = False,
    global_step: int = 0,
):
    """优化的训练循环"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 修复 PyTorch AMP 弃用警告
    if use_amp:
        try:
            # 新版本 PyTorch
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            # 旧版本 PyTorch
            scaler = GradScaler()
    else:
        scaler = None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
    optimizer.zero_grad()
    
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        clean = batch['clean'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)  # [B, H, W] - padding mask

        # Forward pass
        # 预训练时 condition=None，模型学习从噪声恢复原始图像
        # 条件生成时 condition 不为 None，模型学习基于条件生成目标图像
        # 传入 mask 以支持 padding mask
        model_ref = model.module if hasattr(model, 'module') else model
        
        # 获取 condition（如果存在）
        condition = None
        if 'condition' in batch:
            condition_img = batch['condition'].to(device, non_blocking=True)  # [B, C, H_cond, W_cond]
            # 将 condition 图像转换为 patches
            condition = model_ref.image_to_patches(condition_img)  # [B, cond_patches, patch_dim]
        
        B = clean.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, 1000, (B,), device=device)
        
        # Add noise to clean image
        noisy_target, _ = add_noise_to_timestep(clean, t)
        
        # Forward pass with mixed precision
        # 修复 PyTorch AMP 弃用警告
        if use_amp:
            try:
                # 新版本 PyTorch
                amp_context = torch.amp.autocast('cuda')
            except AttributeError:
                # 旧版本 PyTorch
                amp_context = autocast(enabled=True)
        else:
            class NoOpContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            amp_context = NoOpContext()
        
        with amp_context:
            clean_pred = model(noisy_target, t, condition=condition, mask=mask)
            
            # 数值稳定性检查：检查模型输出
            if torch.isnan(clean_pred).any() or torch.isinf(clean_pred).any():
                if rank == 0:
                    print(f"Warning: model output contains nan/inf at batch {batch_idx}, skipping...")
                continue
            
            clean_pred_img = model_ref.patches_to_image(clean_pred)
            
            # 数值稳定性检查：检查图像
            if torch.isnan(clean_pred_img).any() or torch.isinf(clean_pred_img).any():
                if rank == 0:
                    print(f"Warning: clean_pred_img contains nan/inf at batch {batch_idx}, skipping...")
                continue
            
            # Loss: MSE between predicted and clean image
            # 只对有效像素计算 loss（mask 掉 padding）
            # mask: [B, H, W] -> [B, 1, H, W] for broadcasting
            mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
            
            # 确保 mask 中有有效像素
            mask_sum = mask_expanded.sum()
            if mask_sum < 1e-6:
                if rank == 0:
                    print(f"Warning: all pixels are padding at batch {batch_idx}, skipping...")
                continue
            
            # 计算 masked loss
            diff = (clean_pred_img - clean) ** 2
            masked_diff = diff * mask_expanded
            loss = masked_diff.sum() / mask_sum  # 归一化
            loss = loss / gradient_accumulation_steps
            
            # 数值稳定性检查：检查 loss
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"Warning: loss is nan/inf at batch {batch_idx}, skipping...")
                continue
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                # Gradient clipping
                scaler.unscale_(optimizer)
                
                # 检查梯度是否包含 nan/inf
                grad_norm = 0.0
                has_nan_inf = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            if rank == 0 and not has_nan_inf:
                                print(f"Warning: gradient contains nan/inf in {name}, zeroing...")
                            param.grad.zero_()
                            has_nan_inf = True
                        else:
                            param_norm = param.grad.data.norm(2)
                            grad_norm += param_norm.item() ** 2
                
                if not has_nan_inf:
                    grad_norm = grad_norm ** (1. / 2)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 检查梯度是否包含 nan/inf
                has_nan_inf = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            if rank == 0 and not has_nan_inf:
                                print(f"Warning: gradient contains nan/inf in {name}, zeroing...")
                            param.grad.zero_()
                            has_nan_inf = True
                
                if not has_nan_inf:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            # Update learning rate scheduler (每个优化步骤后更新)
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
        
        # 只有在 loss 有效时才更新统计信息
        if not (torch.isnan(loss) or torch.isinf(loss)):
            # Log metrics to wandb
            if rank == 0 and use_wandb and WANDB_AVAILABLE:
                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                loss_value = loss.item() * gradient_accumulation_steps
                
                # Calculate gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                wandb.log({
                    'train/loss': loss_value,
                    'train/learning_rate': current_lr,
                    'train/grad_norm': total_norm,
                    'train/step': global_step,
                }, step=global_step)
                
                global_step += 1
        
        # 只有在 loss 有效时才更新统计信息
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        if rank == 0:
                pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if rank == 0 and use_wandb and WANDB_AVAILABLE:
        epoch_time = time.time() - epoch_start_time
        wandb.log({
            'train/epoch_loss': avg_loss,
            'train/epoch_time': epoch_time,
            'train/samples_per_second': len(dataloader) * dataloader.batch_size / epoch_time,
            'train/epoch': epoch,
        }, step=global_step)
    
    return avg_loss, global_step


def main():
    parser = argparse.ArgumentParser(description='Optimized Training for JiT-based Token2Img Diffusion Model')
    
    # Model args
    parser.add_argument('--model', type=str, default='JiT-B/4', help='Model name (JiT-B/4 for 64×64, JiT-B/16 for 256×256)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--predict_clean', action='store_true', default=True, help='Predict clean data')
    
    # Data args
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--tokenizer_path', type=str, default='/root/data/AI/pretrain/Qwen2.5-7B-Instruct', help='Tokenizer path')
    parser.add_argument('--use_chat_template', action='store_true', help='Use chat_template for QA pairs (for fine-tuning)')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens per image (None = use image capacity)')
    parser.add_argument('--enable_condition', action='store_true', help='Enable conditional generation (requires prompt/answer pairs in data)')
    parser.add_argument('--cond_img_size', type=int, default=None, help='Condition image size (default: 64 if enable_condition=True)')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    # Optimization args
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    
    # Other args
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval (epochs)')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval (batches)')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='jit-diffusion', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity/username (default: use wandb login settings)')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
        print(f"Using mixed precision: {args.use_amp}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        
        # Initialize wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_kwargs = {
                'project': args.wandb_project,
                'name': args.wandb_name or f"jit-{args.model}-{args.img_size}",
                'config': {
                    'model': args.model,
                    'img_size': args.img_size,
                    'batch_size': args.batch_size,
                    'gradient_accumulation_steps': args.gradient_accumulation_steps,
                    'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'warmup_epochs': args.warmup_epochs,
                    'use_amp': args.use_amp,
                    'max_grad_norm': args.max_grad_norm,
                    'enable_condition': args.enable_condition,
                    'cond_img_size': args.cond_img_size,
                },
                'dir': args.output_dir,
            }
            # 如果指定了 entity，添加到 kwargs
            if args.wandb_entity:
                wandb_kwargs['entity'] = args.wandb_entity
            
            wandb.init(**wandb_kwargs)
            entity_info = f", entity={args.wandb_entity}" if args.wandb_entity else ""
            print(f"Wandb initialized: project={args.wandb_project}, name={args.wandb_name or f'jit-{args.model}-{args.img_size}'}{entity_info}")
        elif args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: --use_wandb specified but wandb not available. Install with: pip install wandb")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # Build model
    model = build_jit_model(
        model_name=args.model,
        img_size=args.img_size,
        predict_clean=args.predict_clean,
    )
    model = model.to(device)
    
    # Distributed model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Dataset and dataloader
    dataset = TokenImageDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        img_size=args.img_size,
        use_chat_template=args.use_chat_template,
        max_tokens=args.max_tokens,
        enable_condition=args.enable_condition,
        cond_img_size=args.cond_img_size,
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler
    # 考虑梯度累积，实际优化步数 = len(dataloader) / gradient_accumulation_steps * epochs
    effective_batches_per_epoch = max(1, len(dataloader) // args.gradient_accumulation_steps)
    total_steps = effective_batches_per_epoch * args.epochs
    warmup_steps = effective_batches_per_epoch * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay after warmup
            if total_steps > warmup_steps:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.0, 0.5 * (1 + math.cos(progress * math.pi)))
            else:
                return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        # Train
        avg_loss, global_step = train_epoch_optimized(
            model, dataloader, optimizer, device, epoch, rank,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            scheduler=scheduler,  # 传入 scheduler 以便在训练循环中更新
            use_wandb=args.use_wandb,
            global_step=global_step,
        )
        
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")
            
            # 每 log_interval 个 epoch 记录一次详细信息
            if epoch % args.log_interval == 0:
                print(f"  - Effective batches per epoch: {effective_batches_per_epoch}")
                print(f"  - Total optimization steps: {total_steps}")
                print(f"  - Warmup steps: {warmup_steps}")
                print(f"  - Current step: {scheduler.last_epoch if scheduler else 0}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, 'best_model.pth')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, best_path)
                print(f"Saved best model (loss={avg_loss:.4f}) to {best_path}")
            
            # Log best loss to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({'train/best_loss': best_loss}, step=global_step)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
