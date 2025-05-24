import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


# Original DiffusionPrior class remains commented out
# ...existing code...

# class DiffusionPriorUNet(nn.Module):

#     def __init__(
#             self, 
#             in_channels=4,  # Changed from embed_dim to in_channels for image data
#             cond_channels=4,  # Changed from cond_dim to cond_channels
#             hidden_dim=[64, 128, 256, 512, 1024],  # Reversed order for conv architecture
#             time_embed_dim=512,
#             act_fn=nn.SiLU,
#             dropout=0.0,
#             image_size=64,  # Added image size parameter
#         ):
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.cond_channels = cond_channels
#         self.hidden_dim = hidden_dim
#         self.image_size = image_size

#         # 1. time embedding
#         self.time_proj = Timesteps(time_embed_dim, True, 0)

#         # 2. conditional embedding pathway (changed to Conv layers)
#         self.cond_encoder = nn.Sequential(
#             nn.Conv2d(cond_channels, hidden_dim[0], kernel_size=3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=3, padding=1),
#         )

#         # 3. UNet architecture

#         # 3.1 input layer
#         self.input_layer = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim[0], kernel_size=3, padding=1),
#             nn.GroupNorm(8, hidden_dim[0]),
#             act_fn(),
#         )

#         # 3.2 down blocks
#         self.num_layers = len(hidden_dim)
#         self.encode_time_embedding = nn.ModuleList(
#             [TimestepEmbedding(
#                 time_embed_dim,
#                 hidden_dim[i],
#             ) for i in range(self.num_layers-1)]
#         )
        
#         # Replace linear layers with Conv2d for spatial data
#         self.encode_layers = nn.ModuleList([
#             nn.Module() for _ in range(self.num_layers-1)
#         ])
        
#         for i in range(self.num_layers-1):
#             self.encode_layers[i] = nn.Sequential(
#                 nn.Conv2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, padding=1, stride=2),  # downsampling
#                 nn.GroupNorm(8, hidden_dim[i+1]),
#                 act_fn(),
#                 nn.Dropout(dropout),
#                 nn.Conv2d(hidden_dim[i+1], hidden_dim[i+1], kernel_size=3, padding=1),
#                 nn.GroupNorm(8, hidden_dim[i+1]),
#                 act_fn(),
#                 nn.Dropout(dropout),
#             )

#         # 3.3 up blocks with skip connections
#         self.decode_time_embedding = nn.ModuleList(
#             [TimestepEmbedding(
#                 time_embed_dim,
#                 hidden_dim[i],
#             ) for i in range(self.num_layers-1, 0, -1)]
#         )
        
#         # Replace linear layers with ConvTranspose2d for upsampling
#         self.decode_layers = nn.ModuleList([
#             nn.Module() for _ in range(self.num_layers-1)
#         ])
        
#         for i in range(self.num_layers-1):
#             self.decode_layers[i] = nn.Sequential(
#                 nn.Conv2d(hidden_dim[self.num_layers-1-i], hidden_dim[self.num_layers-1-i], kernel_size=3, padding=1),
#                 nn.GroupNorm(8, hidden_dim[self.num_layers-1-i]),
#                 act_fn(),
#                 nn.Dropout(dropout),
#                 nn.ConvTranspose2d(hidden_dim[self.num_layers-1-i], hidden_dim[self.num_layers-2-i], 
#                                   kernel_size=4, stride=2, padding=1),  # upsampling
#                 nn.GroupNorm(8, hidden_dim[self.num_layers-2-i]),
#                 act_fn(),
#                 nn.Dropout(dropout),
#             )

#         # 3.4 output layer
#         self.output_layer = nn.Sequential(
#             nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=3, padding=1),
#             nn.GroupNorm(8, hidden_dim[0]),
#             act_fn(),
#             nn.Conv2d(hidden_dim[0], in_channels, kernel_size=3, padding=1)
#         )
        

#     def forward(self, x, t, c=None):
#         # x: (batch_size, in_channels, H, W)
#         # t: (batch_size, )
#         # c: (batch_size, cond_channels, H, W)

#         # 1. time embedding
#         t_emb = self.time_proj(t)  # (batch_size, time_embed_dim)

#         # 2. process conditional input if provided
#         cond_features = None
#         if c is not None:
#             cond_features = self.cond_encoder(c)

#         # 3. UNet architecture

#         # 3.1 input
#         x = self.input_layer(x)
#         if cond_features is not None:
#             x = x + cond_features

#         # 3.2 encoder path (down)
#         skip_connections = []
#         for i in range(self.num_layers-1):
#             skip_connections.append(x)
#             # Add time embedding as channel-wise information
#             t_emb_i = self.encode_time_embedding[i](t_emb)
#             # Reshape time embedding to add to all spatial locations
#             t_emb_i = t_emb_i.unsqueeze(-1).unsqueeze(-1)
#             x = x + t_emb_i
#             # Apply convolutions and downsample
#             x = self.encode_layers[i](x)
        
#         # 3.3 decoder path (up) with skip connections
#         for i in range(self.num_layers-1):
#             # Add time embedding
#             t_emb_i = self.decode_time_embedding[i](t_emb)
#             t_emb_i = t_emb_i.unsqueeze(-1).unsqueeze(-1)
#             x = x + t_emb_i
#             # Upsample and apply convolutions
#             x = self.decode_layers[i](x)
#             # Add skip connection
#             x = x + skip_connections[-(i+1)]
            
#         # 3.4 output projection
#         x = self.output_layer(x)

#         return x
class DiffusionPriorUNet(nn.Module):

    def __init__(
            self, 
            in_channels=4,
            cond_channels=4,
            hidden_dim=[64, 128, 256, 512, 1024],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
            image_size=64,
        ):
        super().__init__()
        
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.num_layers = len(hidden_dim)

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. Initial conditional embedding
        self.cond_initial = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_dim[0], kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim[0]),
            act_fn(),
        )

        # 3. UNet architecture

        # 3.1 input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim[0], kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim[0]),
            act_fn(),
        )

        # 3.2 down blocks
        self.encode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i]) 
            for i in range(self.num_layers-1)
        ])
        
        # Encoder blocks - each one reduces spatial dimensions
        self.encode_blocks = nn.ModuleList()
        
        for i in range(self.num_layers-1):
            self.encode_blocks.append(nn.Sequential(
                nn.Conv2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, padding=1, stride=2),
                nn.GroupNorm(8, hidden_dim[i+1]),
                act_fn(),
                nn.Dropout(dropout),
                nn.Conv2d(hidden_dim[i+1], hidden_dim[i+1], kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim[i+1]),
                act_fn(),
                nn.Dropout(dropout),
            ))
        
        # Condition downsamplers
        self.cond_downsamplers = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.cond_downsamplers.append(nn.Sequential(
                nn.Conv2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, padding=1, stride=2),
                nn.GroupNorm(8, hidden_dim[i+1]),
                act_fn(),
            ))

        # 3.3 up blocks with skip connections
        self.decode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i]) 
            for i in range(self.num_layers-1, 0, -1)
        ])
        
        # Decoder blocks - each one increases spatial dimensions
        self.decode_blocks = nn.ModuleList()
        
        for i in range(self.num_layers-1):
            # Current level is num_layers-1-i
            # Next level is num_layers-2-i
            curr_level = self.num_layers-1-i
            next_level = self.num_layers-2-i
            
            self.decode_blocks.append(nn.Sequential(
                nn.Conv2d(hidden_dim[curr_level], hidden_dim[curr_level], kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim[curr_level]),
                act_fn(),
                nn.Dropout(dropout),
                nn.ConvTranspose2d(hidden_dim[curr_level], hidden_dim[next_level], 
                                  kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim[next_level]),
                act_fn(),
                nn.Dropout(dropout),
            ))

        # 3.4 output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim[0]),
            act_fn(),
            nn.Conv2d(hidden_dim[0], in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t, c=None):
        # x: (batch_size, in_channels, H, W)
        # t: (batch_size, )
        # c: (batch_size, cond_channels, H, W)

        # 1. time embedding
        t_emb = self.time_proj(t)  # (batch_size, time_embed_dim)

        # 2. process conditional input
        cond_features = []
        if c is not None:
            # Process the initial conditioning
            cond = self.cond_initial(c)
            cond_features.append(cond)
            
            # Generate conditioning at different resolutions
            for i in range(self.num_layers-1):
                cond = self.cond_downsamplers[i](cond)
                cond_features.append(cond)
        else:
            # No conditioning - use zeros or None
            cond_features = [None] * self.num_layers

        # 3. UNet architecture

        # 3.1 input
        x = self.input_layer(x)
        
        # Apply first level conditioning if available
        if cond_features[0] is not None:
            x = x + cond_features[0]

        # 3.2 encoder path (down) with skip connections
        skip_connections = []
        for i in range(self.num_layers-1):
            # Store for skip connection
            skip_connections.append(x)
            
            # Add time embedding
            t_emb_i = self.encode_time_embedding[i](t_emb)
            t_emb_i = t_emb_i.unsqueeze(-1).unsqueeze(-1)
            x = x + t_emb_i
            
            # Apply convolution block (decreases resolution)
            x = self.encode_blocks[i](x)
            
            # Add conditioning at this resolution level
            if cond_features[i+1] is not None:
                x = x + cond_features[i+1]
        
        # 3.3 decoder path (up) with skip connections
        for i in range(self.num_layers-1):
            # Add time embedding
            t_emb_i = self.decode_time_embedding[i](t_emb)
            t_emb_i = t_emb_i.unsqueeze(-1).unsqueeze(-1)
            x = x + t_emb_i
            
            # Apply convolution and upsampling
            x = self.decode_blocks[i](x)
            
            # Add skip connection from corresponding encoder level
            # Note: num_layers-2-i goes from deep to shallow in the encoder
            x = x + skip_connections[-(i+1)]
            
            # Add conditioning from the corresponding level in the encoder path
            # This matches the skip connection level
            idx = self.num_layers-2-i
            if idx >= 0 and cond_features[idx] is not None:
                x = x + cond_features[idx]
            
        # 3.4 output projection
        x = self.output_layer(x)

        return x

class ImageDataset(Dataset):
    """Dataset for 4×64×64 images and their conditioning images"""
    
    def __init__(self, target_images, condition_images=None):
        """
        Args:
            target_images: Tensor of shape (N, 4, 64, 64)
            condition_images: Optional tensor of shape (N, 4, 64, 64)
        """
        self.target_images = target_images
        self.condition_images = condition_images

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        item = {
            "target_image": self.target_images[idx],
        }
        
        if self.condition_images is not None:
            item["condition_image"] = self.condition_images[idx]
            
        return item


# Keep the original EmbeddingDataset class
# ...existing code...

# Modify the Pipe class to work with images
class Pipe:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
        else:
            self.scheduler = scheduler
            
        self.device = device
        
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss()
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
                c_images = batch['condition_image'].to(device) if 'condition_image' in batch.keys() else None
                target_images = batch['target_image'].to(device)
                N = target_images.shape[0]

                # 1. randomly replacing c_images with None for unconditional training
                if c_images is not None and torch.rand(1) < 0.1:
                    c_images = None

                # 2. Generate noisy images as input
                noise = torch.randn_like(target_images)

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to target images
                noisy_images = self.scheduler.add_noise(
                    target_images,
                    noise,
                    timesteps
                )

                # 5. predict noise
                noise_pred = self.diffusion_prior(noisy_images, timesteps, c_images)
                
                # 6. loss function
                loss = criterion(noise_pred, noise)
                # loss = loss.mean()
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch}')

    def generate(
            self, 
            c_images=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None,
            batch_size=1,
            image_size=64
        ):
        # c_images: (batch_size, cond_channels, H, W)
        self.diffusion_prior.eval()
        
        # Handle batch size and device placement
        if c_images is not None:
            N = c_images.shape[0]
            c_images = c_images.to(self.device)
        else:
            N = batch_size

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare initial noise
        h_t = torch.randn(
            N, 
            self.diffusion_prior.in_channels, 
            image_size, 
            image_size, 
            generator=generator, 
            device=self.device
        )

        # 3. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_batch = torch.ones(N, dtype=torch.long, device=self.device) * t
            
            # 3.1 noise prediction
            if guidance_scale == 0 or c_images is None:
                noise_pred = self.diffusion_prior(h_t, t_batch)
            else:
                # Classifier-free guidance approach
                noise_pred_cond = self.diffusion_prior(h_t, t_batch, c_images)
                noise_pred_uncond = self.diffusion_prior(h_t, t_batch)
                # Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 3.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t.long().item(), h_t, generator=generator).prev_sample
        
        return h_t


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Test the modified model with 4×64×64 images
    prior = DiffusionPriorUNet(in_channels=4, cond_channels=4)
    x = torch.randn(2, 4, 64, 64)  # 2 images with 4 channels, 64×64 resolution
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 4, 64, 64)  # Conditioning images
    y = prior(x, t, c)
    print(y.shape)  # Should output: torch.Size([2, 4, 64, 64])