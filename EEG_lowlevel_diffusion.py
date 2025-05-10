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

from EEG_low_level_encoders import Deconv_EEGConformer,encoder_low_level,encoder_low_level_channelwise,ATMS_Deconv,Config
# from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
# from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
# from subject_layers.Embed import DataEmbedding
# from diffusion_prior import Pipe,EmbeddingDataset
from EEG_diffusion_prior_lowlevel import DiffusionModel4x64x64, Pipe4x64x64, EmbeddingDataset

import argparse
import datetime



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
    parser.add_argument('--model_path', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Generation/models',help='Path to the pre-trained model')
    parser.add_argument('--subject_id', type=str, default='sub-01', help='Subject ID to analyze')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time for analysis window')
    parser.add_argument('--end_time', type=float, default=1.0, help='End time for analysis window')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
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
    parser.add_argument('--latent_mapping', default='VAE', help='Whether to use latent mapping')
    
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
                "latent_mapping": args.latent_mapping
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
    train_dataloader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=False)

    # Set the random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # Initialize and load the pre-trained ATM model
    # Load the pre-trained model
    if config['channels'][0] == 'All':
        n_channels = len(train_d[0][1])
    else:
        n_channels = len(config['channels'])
    if config['model_type']== 'encoder_low_level':
        eeg_model = encoder_low_level(num_channels=n_channels, sequence_length=ntimes).to(device)
    elif config['model_type'] == 'EEGConformer':
        eeg_model = Deconv_EEGConformer(n_outputs=2, n_chans=n_channels, n_filters_time=180, 
                                    filter_time_length=20, pool_time_length=5, pool_time_stride=5, 
                                    drop_prob=0.5, att_depth=3, att_heads=30, att_drop_prob=0.5, 
                                    final_fc_length='auto', return_features=False, n_times=ntimes, 
                                    chs_info=None, input_window_seconds=None, sfreq=None, n_classes=None, 
                                    n_channels=None, input_window_samples=None).to(device)
    elif config['model_type'] == 'encoder_low_level_channelwise':
        eeg_model = encoder_low_level(num_channels=n_channels, sequence_length=ntimes).to(device)
    elif config['model_type'] == 'ATMS':
        ATM_config = Config(seq_len=ntimes,ATMoutput=1024)
        eeg_model=ATMS_Deconv(ATM_config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    #get the number of parameters in the model
    num_params = sum(p.numel() for p in eeg_model.parameters() if p.requires_grad)
    # m_path=config['model_path']
    path_modelstate=f"{config['model_path']}/low_level/{config['model_type']}/{config['subject_id'][0]}/C{n_channels}-{config['time_window'][1]}s-avg{config['average_eeg']}"
    matching_files = [f for f in os.listdir(path_modelstate) if f.startswith('model')][0]
    model_path = os.path.join(path_modelstate, matching_files)
    
    # Load the model state
    checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    eeg_model.load_state_dict(checkpoint)
    eeg_model.to(device)
    eeg_model.eval()
    print(f"Model loaded from {model_path}")
    ### Obtain the EEG features
    image_embed_train,eeg_embed_train=get_eegembed(eeg_model, train_dataloader, device,config['subject_id'][0])

    del train_dataloader,train_d,combined_dataset,validation_d
    torch.cuda.empty_cache()

    dataset_train = EmbeddingDataset(c_embeddings=eeg_embed_train, h_embeddings=image_embed_train)
    dl_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)

    diffusion_prior = DiffusionModel4x64x64()
    # # number of parameters
    # print('number of parameters in the DM:'+sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe4x64x64(diffusion_prior=diffusion_prior,device=device)

    # load pretrained model
    model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'
    pipe.train(dl_train, num_epochs=config['epochs'], learning_rate=config['learning_rate']) # to 0.142     
        # Save the model weights
    model_path=f"./models/DM_lowlevel/{config['subject_id'][0]}"
    os.makedirs(model_path, exist_ok=True)             
    file_path = f"{model_path}/lowlevel_DM.pth"
    torch.save(pipe.diffusion_prior.state_dict(), file_path)
if __name__ == '__main__':
    main()