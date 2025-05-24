import torch
from torch import nn,Tensor
from braindecode.models import EEGConformer
# from ATMS_reconstruction import ATMS
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from einops.layers.torch import Rearrange

from braindecode.models import EEGNetv4


class EEGUpsampler(nn.Module):
    def __init__(self, input_dim=512, initial_channels=256, output_shape=(4, 64, 64)):
        super().__init__()

        # Compute intermediate spatial size (e.g., 4x4)
        self.fc = nn.Linear(input_dim, initial_channels * 4 * 4)

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(initial_channels, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),  # 64x64
            # No activation if you're sending this to a decoder that handles activation (e.g., sigmoid/tanh)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8x8
            nn.Conv2d(initial_channels, 128, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 16x16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 32x32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(32, 4, 3, padding=1),
        )


    def forward(self, x):  # x: [B, D]
        x = self.fc(x)  # [B, C*4*4]
        x = x.view(x.size(0), -1, 4, 4)  # [B, C, 4, 4]
        # x = self.upsample(x)  # [B, 4, 64, 64]
        x = self.upsample2(x)  # [B, 4, 64, 64]
        return x


## This code defines a custom EEGConformer model with a deconvolutional decoder for EEG data.
class EEGConformer_Deconv(EEGConformer):
    def __init__(self, *args, image_size=(4, 64,64) ,**kwargs):
        super().__init__(*args, **kwargs)
        # Remove components you don't need
        del self.final_layer, self.fc
        # Add your custom decoder

        final_fc_length = kwargs.get('final_fc_length', 'auto')
        if final_fc_length == "auto":
            final_fc_length = self.get_fc_size()

        print(f"Size of final fc layer: {final_fc_length}")
        self.deconv = _DeconvDecoder(final_fc_length=final_fc_length,image_size=image_size)
    
    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.deconv(x)
        return x
    
class EEGConformer_Deconv2(EEGConformer):
    def __init__(self, *args, image_size=(4, 64,64) ,**kwargs):
        super().__init__(*args, **kwargs)
        # Remove components you don't need
        del self.final_layer, self.fc
        # Add your custom decoder

        final_fc_length = kwargs.get('final_fc_length', 'auto')
        if final_fc_length == "auto":
            final_fc_length = self.get_fc_size()

        print(f"Size of final fc layer: {final_fc_length}")
        self.deconv = EEGUpsampler(input_dim=final_fc_length, initial_channels=256)
    
    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.deconv(x)
        return x

class _DeconvDecoder(nn.Module):
    def __init__(self, final_fc_length,  image_size=(4, 64, 64)):
        super().__init__()
        print(image_size)
        self.image_size = image_size
        # Reshape to initial feature maps (4x4 with 512 channels)
        self.initial_size = 4
        self.reshape_features = 512

        reshape_total = self.reshape_features * self.initial_size * self.initial_size
        # Initial fully connected layer to convert from EEG features to features suitable for deconvolution
        self.linear = nn.Sequential(
            nn.Linear(final_fc_length, reshape_total)     
        )
        # Progressive deconvolution/upsampling layers
        self.deconv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(self.reshape_features , 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            
            # Final layer to get 4 channels
            nn.ConvTranspose2d(32, 4, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x: Tensor) -> Tensor:
        # Reshape the input tensor
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, self.reshape_features, self.initial_size, self.initial_size)
        # Pass through the deconvolutional layers
        x = self.deconv_layers(x)
        # # Reshape to the desired output size
        # x = x.view(batch_size, 4, self.image_size[1], self.image_size[2])
        return x
    
class _DeconvDecoder_conv(nn.Module):
    def __init__(self, final_fc_length, image_size=(4, 64, 64)):
        super().__init__()
        print(image_size)
        self.image_size = image_size
        # Reshape to initial feature maps (4x4 with 512 channels)
        self.initial_size = 4
        self.reshape_features = 512

        reshape_total = self.reshape_features * self.initial_size * self.initial_size
        # Initial fully connected layer to convert from EEG features to features suitable for deconvolution
        self.linear = nn.Sequential(
            nn.Linear(final_fc_length, reshape_total)     
        )
        
        # Initial convolution block
        self.init_conv_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU()
        )
        
        # First upsampling block: 4x4 -> 8x8
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU()
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        
        # Second upsampling block: 8x8 -> 16x16
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        
        # Third upsampling block: 16x16 -> 32x32
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        
        # Fourth upsampling block: 32x32 -> 64x64
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        
        # Final refinement and output
        self.final_conv_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        
        # Final output layer
        self.final_layer = nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: Tensor) -> Tensor:
        # Reshape the input tensor
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, self.reshape_features, self.initial_size, self.initial_size)
        
        # Initial convolution at 4x4
        x = self.init_conv_block(x)
        
        # First upsampling block: 4x4 -> 8x8
        x = self.conv_block1(x)
        x = self.upsample1(x)
        
        # Second upsampling block: 8x8 -> 16x16
        x = self.conv_block2(x)
        x = self.upsample2(x)
        
        # Third upsampling block: 16x16 -> 32x32
        x = self.conv_block3(x)
        x = self.upsample3(x)
        
        # Fourth upsampling block: 32x32 -> 64x64
        x = self.conv_block4(x)
        x = self.upsample4(x)
        
        # Final refinement and output
        x = self.final_conv_block(x)
        x = self.final_layer(x)
        
        return x
    
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.GroupNorm(8, channels),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.GroupNorm(8, channels),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, 3, padding=1),
#         )

#     def forward(self, x):
#         return x + self.block(x)
    
# class SelfAttention(nn.Module):
#     def __init__(self, channels, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads
#         assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        
#         self.qkv = nn.Conv2d(channels, channels * 3, 1)
#         self.proj = nn.Conv2d(channels, channels, 1)
        
#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # Generate query, key, value projections
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, H*W, head_dim)
        
#         # Multi-head attention
#         attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, H*W, H*W)
#         attn = attn.softmax(dim=-1)
        
#         # Apply attention weights
#         out = attn @ v  # (B, num_heads, H*W, head_dim)
        
#         # Reshape and project back
#         out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
#         out = self.proj(out)
        
#         return x + out
    
# class UpsampleBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upsample = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         )

#     def forward(self, x):
#         return self.upsample(x)


# class _Res_attention_Decon(nn.Module):
#     def __init__(self,input_dim=1024):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, 256 * 4 * 4)

#         self.res1 = ResBlock(256)
#         self.attn1 = SelfAttention(256)
#         self.up1 = UpsampleBlock(256, 128)  # 4→8

#         self.res2 = ResBlock(128)
#         self.up2 = UpsampleBlock(128, 128)  # 8→16

#         self.res3 = ResBlock(128)
#         self.attn3 = SelfAttention(128)
#         self.up3 = UpsampleBlock(128, 64)   # 16→32

#         self.res4 = ResBlock(64)
#         self.up4 = UpsampleBlock(64, 32)    # 32→64

#         self.out_conv = nn.Conv2d(32, 4, kernel_size=3, padding=1)

#     def forward(self, z):  # z shape: (B, 1024)
#         x = self.fc(z).view(-1, 256, 4, 4)

#         x = self.res1(x)
#         x = self.attn1(x)
#         x = self.up1(x)

#         x = self.res2(x)
#         x = self.up2(x)

#         x = self.res3(x)
#         x = self.attn3(x)
#         x = self.up3(x)

#         x = self.res4(x)
#         x = self.up4(x)

#         x = self.out_conv(x)  # → (B, 4, 64, 64)
#         return x

    


### Low-level encoder for EEG data in the ATM paper
class encoder_low_level(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1):
        super().__init__()        
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, 128) for _ in range(num_subjects)])

        # CNN upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),    # Keep size (64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),    # Output shape (4, 64, 64)
        )


    def forward(self, x):
        # Apply subject-wise linear layer
        x = self.subject_wise_linear[0](x)  # Output shape: (batchsize, 63, 128)
        # Reshape to match the input size for the upsampler
        x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
        out = self.upsampler(x)  # Pass through the upsampler
        return out
    
class encoder_low_level_channelwise(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250):
        super().__init__()        
        self.channel_wise_linear = nn.ModuleList([nn.Linear(sequence_length, 128) for _ in range(num_channels)])

        # CNN upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),    # Keep size (64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),    # Output shape (4, 64, 64)
        )


    def forward(self, x):
        # Apply subject-wise linear layer
        # Apply channel-specific linear layers
        outputs = []
        for i in range(x.size(1)):  # Loop through each channel
            channel_data = x[:, i, :]  # Extract single channel data: (batch, sequence_length)
            transformed = self.channel_wise_linear[i](channel_data)  # Apply that channel's specific linear layer
            outputs.append(transformed)
        x = torch.stack(outputs, dim=1)  # Recombine: (batch, num_channels, 128)
        
        # Reshape to match the input size for the upsampler
        x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
        out = self.upsampler(x)  # Pass through the upsampler
        return out
class Config:
    def __init__(self,ATMoutput=1024,seq_len=250,enc_in=63):
        self.task_name = 'classification'  # Example task name
        self.seq_len = seq_len                # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = enc_in                   # Encoder input dimension (example value)
        self.ATMout=ATMoutput

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]      
        # print("enc_out", enc_out.shape)
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS(nn.Module):    
    def __init__(self, config):
        super(ATMS, self).__init__()
        default_config = config
        self.encoder = iTransformer(default_config)   
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg(proj_dim=default_config.ATMout)        
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out  


class ATMS_Deconv(nn.Module):    
    def __init__(self,config):
        super(ATMS_Deconv, self).__init__()
        self.encoder=ATMS(config)

        self.deconv = _DeconvDecoder(final_fc_length=config.ATMout,image_size=(4, 64, 64))
      
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, subject_ids)
        out = self.deconv(x)
        return out
    
# class ATMS_Res_attention(nn.Module):    
#     def __init__(self,config):
#         super(ATMS_Res_attention, self).__init__()
#         self.encoder=ATMS(config)

#         self.deconv = _Res_attention_Decon(input_dim=1024)
      
         
#     def forward(self, x, subject_ids):
#         x = self.encoder(x, subject_ids)
#         out = self.deconv(x)
#         return out
    

# class ATMS_upsample(nn.Module):    
#     def __init__(self,config):
#         super().__init__()
#         self.encoder=ATMS(config)

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),    # Keep size (64, 64)
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),    # Output shape (4, 64, 64)
#         )
      
         
#     def forward(self, x, subject_ids):
#         x = self.encoder(x, subject_ids)
#         x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
#         out = self.deconv(x)
#         return out
    

