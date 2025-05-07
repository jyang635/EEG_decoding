import os

import torch

from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
# from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
# import csv
from torch import Tensor
# import itertools
# import math
import re

import numpy as np
from loss import ClipLoss
# import argparse
from torch import nn
from torch.optim import AdamW
from ATMS_reconstruction import ATMS

# from diffusion_prior import DiffusionPriorUNet, Pipe, EmbeddingDataset
from full_condition_diffusion_prior import DiffusionPriorUNet, Pipe, EmbeddingDataset


class Config:
    def __init__(self):
        self.task_name = 'Generation'  # Example task name
        self.seq_len = 250                 # Sequence length
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
        self.enc_in = 63                   # Encoder input dimension (example value)



def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def get_eegfeatures(sub, eegmodel, dataloader, device,save_features=True):
    eegmodel.eval()

    features_list = []  # List to store features   
    image_list = [] 
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
      
            eeg_features = eegmodel(eeg_data, subject_ids)
            features_list.append(eeg_features)  # Append the features to the list
            image_list.append(img)  # Append the features to the list
            
        features_tensor = torch.cat(features_list, dim=0)
        print("features_tensor", features_tensor.shape)
        if save_features:
            torch.save(features_tensor.cpu(), f"ATM_S_test_eeg_features_{sub}.pt")  # Save features as .pt file

    return features_tensor.cpu()

def main():
    config = {
    "data_path": "/home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz",
    "project": "train_pos_img_text_rep",
    "entity": "sustech_rethinkingbci",
    "name": "lr=3e-4_img_pos_pro_eeg",
    "lr": 3e-4,
    "epochs": 50,
    "batch_size": 1024,
    "logger": True,
    "encoder_type":'ATMS',
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eeg_model = ATMS(63, 250)
    print('number of parameters:', sum([p.numel() for p in eeg_model.parameters()]))

    #####################################################################################

    # eeg_model.load_state_dict(torch.load("/home/ldy/Workspace/Reconstruction/models/contrast/sub-08/01-30_00-44/40.pth"))
    eeg_model.load_state_dict(torch.load("models/contrast/ATMS/sub-08/04-21_21-50/40.pth",weights_only=True))
    eeg_model = eeg_model.to(device)
    sub = 'sub-08'

    ### Obtain the EEG embedding of the training set

    train_dataset = EEGDataset(config['data_path'], subjects= [sub], train=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    eeg_features_train = get_eegfeatures(sub, eeg_model, train_loader, device,save_features=False)
    print(f"EEG features shape: {eeg_features_train.shape}")
    del eeg_model
    torch.cuda.empty_cache()

    emb_img_train = torch.load('./ViT-H-14_features_train.pt')['img_features']
    emb_img_train_4 = emb_img_train.view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
    #### Train the diffusion prior that mapps the EEG emb to the IMG emb

    dataset = EmbeddingDataset(
        c_embeddings=eeg_features_train, h_embeddings=emb_img_train_4, 
        # h_embeds_uncond=h_embeds_imgnet
    )
    dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    # number of parameters
    print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device=device)

    # load pretrained model
    model_name = 'inhouse_con_diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'

    pipe.train(dl, num_epochs=150, learning_rate=1e-3) # to 0.142 
    sub = 'sub-08'
    # Create proper directory structure for saving the model
    save_path = f'./fintune_ckpts/ATMS/{sub}/{model_name}.pt'
    directory = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    torch.save(pipe.diffusion_prior.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()