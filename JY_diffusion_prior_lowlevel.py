import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):

    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

class DiffusionModel4x64x64(nn.Module):
    """Diffusion model that processes and generates 4×64×64 image data"""
    
    def __init__(
            self,
            channels=4,
            image_size=(64, 64),
            base_channels=64,
            channel_multipliers=(1, 2, 4, 8),
            time_embed_dim=256,
            dropout=0.1,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            use_scale_shift_norm=True
        ):
        super().__init__()
        
        self.channels = channels
        self.image_size = image_size
        
        # Time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        time_embed_dim_out = time_embed_dim * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim_out),
            nn.SiLU(),
            nn.Linear(time_embed_dim_out, time_embed_dim_out),
        )
        
        # Conditioning branch (processes 4×64×64 input)
        self.cond_encoder = nn.ModuleList([
            nn.Conv2d(channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        ])
        
        # Down blocks for conditioning
        curr_res = image_size[0]
        in_channels = base_channels
        feature_maps = []
        
        for multiplier in channel_multipliers:
            out_channels = base_channels * multiplier
            
            for _ in range(num_res_blocks):
                self.cond_encoder.append(ResBlock(
                    in_channels, 
                    out_channels, 
                    time_embed_dim=time_embed_dim_out, 
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm
                ))
                in_channels = out_channels
                if curr_res in attention_resolutions:
                    self.cond_encoder.append(AttentionBlock(in_channels))
                feature_maps.append((curr_res, in_channels))
            
            if multiplier != channel_multipliers[-1]:
                self.cond_encoder.append(Downsample(in_channels))
                curr_res //= 2
                feature_maps.append((curr_res, in_channels))
        
        # Middle block for conditioning
        self.cond_middle = nn.ModuleList([
            ResBlock(in_channels, in_channels, time_embed_dim=time_embed_dim_out, dropout=dropout),
            AttentionBlock(in_channels),
            ResBlock(in_channels, in_channels, time_embed_dim=time_embed_dim_out, dropout=dropout),
        ])
        
        # Main branch (processes input noise to predict output)
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(channels, base_channels, kernel_size=3, padding=1)
        ])
        
        # Down blocks
        curr_res = image_size[0]
        in_channels = base_channels
        feature_maps = []
        
        for multiplier in channel_multipliers:
            out_channels = base_channels * multiplier
            
            for _ in range(num_res_blocks):
                self.input_blocks.append(ResBlock(
                    in_channels, 
                    out_channels, 
                    time_embed_dim=time_embed_dim_out, 
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm
                ))
                in_channels = out_channels
                if curr_res in attention_resolutions:
                    self.input_blocks.append(AttentionBlock(in_channels))
                feature_maps.append((curr_res, in_channels))
            
            if multiplier != channel_multipliers[-1]:
                self.input_blocks.append(Downsample(in_channels))
                curr_res //= 2
                feature_maps.append((curr_res, in_channels))
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResBlock(in_channels, in_channels, time_embed_dim=time_embed_dim_out, dropout=dropout),
            AttentionBlock(in_channels),
            ResBlock(in_channels, in_channels, time_embed_dim=time_embed_dim_out, dropout=dropout),
        ])
        
        # Up blocks
        self.output_blocks = nn.ModuleList([])
        
        for i, multiplier in enumerate(reversed(channel_multipliers)):
            out_channels = base_channels * multiplier
            
            for j in range(num_res_blocks + 1):
                skip_channels = feature_maps.pop()[1]
                self.output_blocks.append(ResBlock(
                    in_channels + skip_channels,
                    out_channels,
                    time_embed_dim=time_embed_dim_out,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm
                ))
                in_channels = out_channels
                if curr_res in attention_resolutions:
                    self.output_blocks.append(AttentionBlock(in_channels))
                
                if i < len(channel_multipliers) - 1 and j == num_res_blocks:
                    self.output_blocks.append(Upsample(in_channels))
                    curr_res *= 2
        
        # Final output
        self.out = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, t, cond=None):
        """
        x: (B, 4, 64, 64) - Noisy input image
        t: (B,) - Timesteps
        cond: (B, 4, 64, 64) - Conditioning image
        """
        # Time embedding
        t_emb = self.time_proj(t)
        t_emb = self.time_embedding(t_emb)
        
        # Process conditioning image if provided
        cond_features = None
        if cond is not None:
            h = cond
            for module in self.cond_encoder:
                h = module(h, t_emb) if isinstance(module, ResBlock) else module(h)
            
            for module in self.cond_middle:
                h = module(h, t_emb) if isinstance(module, ResBlock) else module(h)
                
            cond_features = h
        
        # Process input with conditioning
        h = x
        skips = []
        
        # Down path
        for module in self.input_blocks:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            skips.append(h)
        
        # Add conditioning features at the bottleneck
        if cond_features is not None:
            h = h + cond_features
            
        # Middle
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Up path with skip connections
        for module in self.output_blocks:
            if isinstance(module, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = module(h, t_emb)
            elif isinstance(module, Upsample):
                h = module(h)
            else:
                h = module(h)
        
        # Final output
        return self.out(h)


# Helper modules for the diffusion model
class ResBlock(nn.Module):
    """Residual block with time conditioning"""
    
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout, use_scale_shift_norm=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels * (2 if use_scale_shift_norm else 1))
        self.use_scale_shift_norm = use_scale_shift_norm
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        # Time embedding
        time_emb = self.time_emb_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
            h = self.act2(h)
        else:
            h = self.act2(self.norm2(h + time_emb))
            
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(b, c, -1).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, -1)  # (b, c, hw)
        v = v.reshape(b, c, -1).permute(0, 2, 1)  # (b, hw, c)
        
        # Attention
        scale = 1.0 / math.sqrt(c)
        attn = torch.bmm(q, k) * scale  # (b, hw, hw)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value projection
        h = torch.bmm(attn, v)  # (b, hw, c)
        h = h.permute(0, 2, 1).reshape(b, c, h, w)
        
        return x + self.proj(h)


class Upsample(nn.Module):
    """Upsample module"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    """Downsample module"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)
    
class Pipe4x64x64:
    def __init__(self, diffusion_model=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_model.to(device) if diffusion_model else DiffusionModel4x64x64().to(device)
        
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler()
        else:
            self.scheduler = scheduler
            
        self.device = device
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]

                # 1. randomly replecing c_embeds to None
                if torch.rand(1) < 0.1:
                    c_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(h_embeds)

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to h_embedding
                perturbed_h_embeds = self.scheduler.add_noise(
                    h_embeds,
                    noise,
                    timesteps
                ) # (batch_size, embed_dim), (batch_size, )

                # 5. predict noise
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                
                # 6. loss function weighted by sigma
                loss = criterion(noise_pre, noise) # (batch_size,)
                loss = (loss).mean()
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                lr_scheduler.step()
                optimizer.step()

                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch}')
    def generate(
            self, 
            cond_image=None,
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None
        ):
        """
        Generate a 4×64×64 image conditioned on another 4×64×64 image
        
        Args:
            cond_image: (B, 4, 64, 64) conditioning image
            num_inference_steps: Number of denoising steps
            timesteps: Optional custom timesteps
            guidance_scale: Classifier-free guidance scale
            generator: Random number generator
        """
        self.diffusion_model.eval()
        
        # Prepare conditioning
        if cond_image is not None:
            cond_image = cond_image.to(self.device)
            batch_size = cond_image.shape[0]
        else:
            batch_size = 1
        
        # Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)
        
        # Prepare initial noise
        x_t = torch.randn(batch_size, 4, 64, 64, generator=generator, device=self.device)
        
        # Denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_batch = torch.ones(batch_size, dtype=torch.long, device=self.device) * t
            
            # For classifier-free guidance, run with and without conditioning
            if guidance_scale > 1.0 and cond_image is not None:
                # With conditioning
                with torch.no_grad():
                    noise_pred_cond = self.diffusion_model(x_t, t_batch, cond_image)
                
                # Without conditioning
                with torch.no_grad():
                    noise_pred_uncond = self.diffusion_model(x_t, t_batch, None)
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard inference
                with torch.no_grad():
                    noise_pred = self.diffusion_model(x_t, t_batch, cond_image)
            
            # Update sample with scheduler
            x_t = self.scheduler.step(noise_pred, t.long().item(), x_t, generator=generator).prev_sample
        
        return x_t  # Shape: (B, 4, 64, 64)