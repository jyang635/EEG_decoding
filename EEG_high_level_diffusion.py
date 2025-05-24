import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import DataLoader, Dataset,ConcatDataset

import re

from EEG_ThingsData import load_multiple_subjects

from torch.utils.data import DataLoader, Dataset

from ATMS_reconstruction import ATMS
# from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
# from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
# from subject_layers.Embed import DataEmbedding
from diffusion_prior import DiffusionPriorUNet, Pipe,EmbeddingDataset
import argparse
import datetime

# class Config:
#     def __init__(self):
#         self.task_name = 'classification'  # Example task name
#         self.seq_len = 250                 # Sequence length
#         self.pred_len = 250                # Prediction length
#         self.output_attention = False      # Whether to output attention weights
#         self.d_model = 250                 # Model dimension
#         self.embed = 'timeF'               # Time encoding method
#         self.freq = 'h'                    # Time frequency
#         self.dropout = 0.25                # Dropout rate
#         self.factor = 1                    # Attention scaling factor
#         self.n_heads = 4                   # Number of attention heads
#         self.e_layers = 1                  # Number of encoder layers
#         self.d_ff = 256                    # Dimension of the feedforward network
#         self.activation = 'gelu'           # Activation function
#         self.enc_in = 63                   # Encoder input dimension (example value)

# class iTransformer(nn.Module):
#     def __init__(self, configs, joint_train=False,  num_subjects=10):
#         super(iTransformer, self).__init__()
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#         # Embedding
#         self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads
#                     ),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )

#     def forward(self, x_enc, x_mark_enc, subject_ids=None):
#         # Embedding
#         enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         enc_out = enc_out[:, :63, :]      
#         # print("enc_out", enc_out.shape)
#         return enc_out

# class PatchEmbedding(nn.Module):
#     def __init__(self, emb_size=40):
#         super().__init__()
#         # Revised from ShallowNet
#         self.tsconv = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
#             nn.AvgPool2d((1, 51), (1, 5)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.Dropout(0.5),
#         )

#         self.projection = nn.Sequential(
#             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         # b, _, _, _ = x.shape
#         x = x.unsqueeze(1)     
#         # print("x", x.shape)   
#         x = self.tsconv(x)
#         # print("tsconv", x.shape)   
#         x = self.projection(x)
#         # print("projection", x.shape)  
#         return x

# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x

# class FlattenHead(nn.Sequential):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x = x.contiguous().view(x.size(0), -1)
#         return x

# class Enc_eeg(nn.Sequential):
#     def __init__(self, emb_size=40, **kwargs):
#         super().__init__(
#             PatchEmbedding(emb_size),
#             FlattenHead()
#         )

# class Proj_eeg(nn.Sequential):
#     def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
#         super().__init__(
#             nn.Linear(embedding_dim, proj_dim),
#             ResidualAdd(nn.Sequential(
#                 nn.GELU(),
#                 nn.Linear(proj_dim, proj_dim),
#                 nn.Dropout(drop_proj),
#             )),
#             nn.LayerNorm(proj_dim),
#         )

# class ATMS(nn.Module):    
#     def __init__(self, num_channels=63, sequence_length=25, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
#         super(ATMS, self).__init__()
#         default_config = Config()
#         self.encoder = iTransformer(default_config)   
#         self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
#         self.enc_eeg = Enc_eeg()
#         self.proj_eeg = Proj_eeg()        
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.loss_func = ClipLoss()       
         
#     def forward(self, x, subject_ids):
#         x = self.encoder(x, None, subject_ids)
#         # print(f'After attention shape: {x.shape}')
#         # print("x", x.shape)
#         # x = self.subject_wise_linear[0](x)
#         # print(f'After subject-specific linear transformation shape: {x.shape}')
#         eeg_embedding = self.enc_eeg(x)
        
#         out = self.proj_eeg(eeg_embedding)
#         return out  


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
def get_eegembed(eegmodel, dataloader, device,sub):
    eegmodel.eval()
    eeg_list=[]
    image_list=[]

    with torch.no_grad():
        for batch_idx, (img, eeg_data) in enumerate(dataloader):            
            eeg_data = eeg_data.to(device)
            batch_size = eeg_data.size(0) 
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            eeg_features = eegmodel(eeg_data,subject_ids).float()
            image_list.append(img)
            eeg_list.append(eeg_features)
    
    eeg_list = torch.cat(eeg_list, dim=0)
    image_list = torch.cat(image_list, dim=0)
    # recon_list=positive_images(recon_list)
    print('Successfully extracted EEG features')
    return image_list,eeg_list



def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Conformer+deconv')
    
    # Add your command-line arguments
    parser.add_argument('--eeg_folder', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz')
    parser.add_argument('--img_folder', type=str, default='/home/yjk122/IP_temp/ThingsEEG/image',help='Path to the folder containing images')
    parser.add_argument('--model_path', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Generation/models/contrast/ATMS',help='Path to the pre-trained model')
    parser.add_argument('--subject_id', type=str, default='sub-01', help='Subject ID to analyze')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time for analysis window')
    parser.add_argument('--end_time', type=float, default=1.0, help='End time for analysis window')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--model', type=str,choices=['encoder_low_level', 'encoder_low_level_channelwise', 'EEGConformer','ATMS'], default='ATMS')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--loss', type=str, default='mse', help='loss function to use')
    parser.add_argument('--channels', type=str, default='All', 
                        help='EEG channels to use (comma-separated)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--average_eeg', action='store_true', help='Whether to average EEG data')
    parser.add_argument('--latent_mapping', default='CLIP', help='Whether to use latent mapping')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    
    args = parser.parse_args()
    # Initialize wandb
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    config = {
                "eeg_folder": args.eeg_folder,
                "img_folder": args.img_folder,
                "model_path": args.model_path,
                "model_type": args.model,
                "subject_id": args.subject_id.split(','),
                "time_window": [args.start_time, args.end_time],
                "channels": args.channels.split(','),
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping,
                "seed": args.seed,
                "loss": args.loss,
                'average_eeg': args.average_eeg,
                "latent_mapping": args.latent_mapping,
                "save_model": args.save_model
            }
    # Add this before your training loop
    # print(f"Config: {config['average_eeg']}")

    

    
    EEG_dir = config['eeg_folder']
    img_dir = config['img_folder']
    img_metadata = np.load(os.path.join(img_dir, 'image_metadata.npy'), allow_pickle=True).item()
    train_d,validation_d,ntimes = load_multiple_subjects(subject_ids=config['subject_id'], eeg_dir=EEG_dir, img_dir=img_dir, 
                                                    img_metadata=img_metadata, start_time=config['time_window'][0], 
                                                    end_time=config['time_window'][1],desired_channels=config['channels'],
                                                    image_size=None,compressor=config['latent_mapping'],training=True,average=config['average_eeg'])
    combined_dataset = ConcatDataset([train_d, validation_d])
    train_dataloader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True)

    # Set the random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # Initialize and load the pre-trained ATM model
    eeg_model = ATMS(63, 250)
    model_path4sample=os.path.join(config['model_path'], config['subject_id'][0])
    only_folders = [f for f in os.listdir(model_path4sample) if os.path.isdir(os.path.join(model_path4sample, f))]
    checkpoint= torch.load(os.path.join(model_path4sample, only_folders[0],'40.pth'), map_location=device,weights_only=True)
    eeg_model.load_state_dict(checkpoint)
    eeg_model = eeg_model.to(device)
    ### Obtain the EEG features
    image_embed_train,eeg_embed_train=get_eegembed(eeg_model, train_dataloader, device,config['subject_id'][0])

    del train_dataloader,train_d,combined_dataset

    dataset_train = EmbeddingDataset(c_embeddings=eeg_embed_train, h_embeddings=image_embed_train)
    dl_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    # # number of parameters
    # print('number of parameters in the DM:'+sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device=device)

    # load pretrained model
    model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'
    pipe.train(dl_train, num_epochs=config['epochs'], learning_rate=config['learning_rate']) # to 0.142 
    if config['save_model']:
        # Save the model weights
        model_path=f"./models/DM_highlevel/{config['subject_id'][0]}"
        os.makedirs(model_path, exist_ok=True)             
        file_path = f"{model_path}/highlevel_DM.pth"
        torch.save(pipe.diffusion_prior.state_dict(), file_path)

if __name__ == '__main__':
    main()