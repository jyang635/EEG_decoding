import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,ConcatDataset
import wandb

from tqdm import tqdm
import copy
import re


# from diffusers.utils import load_image

# from diffusers.image_processor import VaeImageProcessor
# from diffusers import AutoencoderKL
from EEG_diffusion_prior_lowlevel import DiffusionPriorUNet

import argparse
import datetime

from EEG_low_level_encoders import EEGConformer_Deconv,encoder_low_level,encoder_low_level_channelwise,ATMS_Deconv,Config,EEGConformer_Deconv2
from EEG_ThingsData import load_multiple_subjects


class ATMS_DM(nn.Module):
    def __init__(self, config):
        super(ATMS_DM, self).__init__()
        self.eeg_model = ATMS_Deconv(config)
        self.diffusion_model = DiffusionPriorUNet(dropout=0.1)

    def forward(self, x, subject_ids,noisy_image,time):
        # Pass EEG data through the EEG model
        if x is not None:
            eeg_features = self.eeg_model(x, subject_ids)
        else:
            eeg_features = None
        # Pass the output through the diffusion model
        diffusion_output = self.diffusion_model(noisy_image,time,eeg_features)
        return diffusion_output
    
    def generate(self, x, subject_ids,scheduler=None,num_inference_steps=50, timesteps=None,guidance_scale=5.0, generator=None,image_size=64,device='cuda'):
        # Pass EEG data through the EEG model
        if x is not None:
            eeg_features = self.eeg_model(x, subject_ids)
        else:
            eeg_features = None
        N = eeg_features.shape[0]
        self.diffusion_model.eval()
        # 1. Prepare timesteps
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            scheduler = DDPMScheduler() 
        else:
            scheduler = scheduler       
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, timesteps)

        # 2. Prepare initial noise
        h_t = torch.randn(
            N, 
            self.diffusion_model.in_channels, 
            image_size, 
            image_size, 
            generator=generator, 
            device=device
        )

        # 3. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_batch = torch.ones(N, dtype=torch.long, device=device) * t
            
            # 3.1 noise prediction
            if guidance_scale == 0 or eeg_features is None:
                noise_pred = self.diffusion_model(h_t, t_batch)
            else:
                # Classifier-free guidance approach
                noise_pred_cond = self.diffusion_model(h_t, t_batch, eeg_features)
                noise_pred_uncond = self.diffusion_model(h_t, t_batch)
                # Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 3.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = scheduler.step(noise_pred, t.long().item(), h_t, generator=generator).prev_sample
        
        return h_t

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def train_loop(dataloader, model, loss_fn, optimizer,lr_scheduler,time_scheduler,device,subject_id=None,model_type='ATMS'):
        model.train()
        loss_sum = 0

        if model_type.startswith('ATMS'):
            for batch, (image,response)in enumerate(dataloader):
                image=image.to(device)
                subject_ids = extract_id_from_string(subject_id)
                eeg_tensor= response.to(device)
                batch_size =eeg_tensor.size(0)
                subject_ids = torch.full((batch_size,), subject_ids, dtype=torch.long).to(device)
                N = image.shape[0]

                # 1. randomly replacing c_images with None for unconditional training
                if torch.rand(1) < 0.1:
                    eeg_tensor = None

                # 2. Generate noisy images as input
                noise = torch.randn_like(image)

                # 3. sample timestep
                timesteps = torch.randint(0, time_scheduler.config.num_train_timesteps, (N,), device=device)

                # 4. add noise to target images
                noisy_images = time_scheduler.add_noise(
                    image,
                    noise,
                    timesteps
                )
                noise_pred = model(eeg_tensor,subject_ids,noisy_images,timesteps)
                loss = loss_fn(noise_pred, noise)
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_sum += loss.item()

        loss_epoch = loss_sum / len(dataloader)
        return loss_epoch
def validation_loop(dataloader, model, loss_fn, optimizer,lr_scheduler,time_scheduler,device,subject_id=None,model_type='ATMS'):
        model.eval()
        loss_sum = 0
        with torch.no_grad():
            if model_type.startswith('ATMS'):
                for batch, (image,response)in enumerate(dataloader):
                    image=image.to(device)
                    subject_ids = extract_id_from_string(subject_id)
                    eeg_tensor= response.to(device)
                    batch_size =eeg_tensor.size(0)
                    subject_ids = torch.full((batch_size,), subject_ids, dtype=torch.long).to(device)
                    N = image.shape[0]

                    # 1. randomly replacing c_images with None for unconditional training
                    if torch.rand(1) < 0.1:
                        eeg_tensor = None

                    # 2. Generate noisy images as input
                    noise = torch.randn_like(image)

                    # 3. sample timestep
                    timesteps = torch.randint(0, time_scheduler.config.num_train_timesteps, (N,), device=device)

                    # 4. add noise to target images
                    noisy_images = time_scheduler.add_noise(
                        image,
                        noise,
                        timesteps
                    )
                    noise_pred = model(eeg_tensor,subject_ids,noisy_images,timesteps)
                    loss = loss_fn(noise_pred, noise)
                    loss_sum += loss.item()

        loss_epoch = loss_sum / len(dataloader)
        return loss_epoch

def main_train_loop(eeg_model, train_dataloader, validation_dataloader,device, 
                     config,model_type='EEGconformer',save_model=False):
    optimizer = torch.optim.AdamW(eeg_model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from diffusers.schedulers import DDPMScheduler
    time_scheduler = DDPMScheduler() 
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * config['epochs']),
    )
    # num_train_timesteps = time_scheduler.config.num_train_timesteps
    eeg_model.to(device)

    subject_ids = config['subject_id']
    best_validation_loss = float('inf')
    stop_counter = 0

    loss_dict = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'ssim': nn.MSELoss()
    }
    loss_fn = loss_dict.get(config['loss'], nn.MSELoss())
    if config['loss'] not in loss_dict.keys():
        print(f"Warning: '{config['loss']}' is not a valid loss function key. Defaulting to 'mse'.")
    else:
        print(f"Using loss function: {config['loss']}")
    for t in range(config['epochs']):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, eeg_model, loss_fn, optimizer,lr_scheduler,time_scheduler,device=device,subject_id=subject_ids[0],model_type=model_type)
        print('Train loss= '+str(round(train_loss, 5)))
        validation_loss = validation_loop(validation_dataloader, eeg_model, loss_fn, optimizer,lr_scheduler,time_scheduler,device=device,subject_id=subject_ids[0],model_type=model_type)
        print('Validation loss= '+str(round(validation_loss, 5)))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = copy.deepcopy(eeg_model.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {stop_counter} epochs without improvement.")
            # if config['save_model']:
            #     print("Saving the best model...")
            #     # Save the best model
            #     model_path=f"./models/Lowlevel_Allin/{config['model_type']}/{config['subject_id'][0]}"
            #     print(f"Saving model to {model_path}")
            #     os.makedirs(model_path, exist_ok=True)             
            #     file_path = f"{model_path}/lowlevel_DM.pth"
            #     torch.save(best_model_state, file_path)
            break
    if config['save_model']:
        # Save the model
        model_path=f"./models/Lowlevel_Allin/{config['model_type']}/{config['subject_id'][0]}"
        print(f"Saving model to {model_path}")
        os.makedirs(model_path, exist_ok=True)             
        file_path = f"{model_path}/lowlevel_DM.pth"
        torch.save(best_model_state, file_path)



def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Conformer+deconv')
    
    # Add your command-line arguments
    parser.add_argument('--eeg_folder', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz')
    parser.add_argument('--img_folder', type=str, default='/home/yjk122/IP_temp/ThingsEEG/image')
    parser.add_argument('--subject_id', type=str, default='sub-01', help='Subject ID to analyze')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time for analysis window')
    parser.add_argument('--end_time', type=float, default=1, help='End time for analysis window')
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size for training')
    parser.add_argument('--model', type=str,choices=['encoder_low_level', 'encoder_low_level_channelwise',
                                                      'EEGConformer','ATMS'], default='ATMS')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--project', type=str, default='ATM_lowlevel', help='W&B project name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--loss', type=str, default='mse', help='loss function to use')
    parser.add_argument('--channels', type=str, default='All', 
                        help='EEG channels to use (comma-separated)')
    parser.add_argument('--image_size', type=str, default="256,256", help='size of the image'),
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--average_eeg', action='store_true', help='Whether to average EEG data')
    parser.add_argument('--latent_mapping', default='VAE', help='Whether to use latent mapping')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model')
    
    args = parser.parse_args()
    # Initialize wandb
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    image_size = args.image_size.split(',')
    image_size = (3,int(image_size[0]), int(image_size[1]))
    config = {
                "eeg_folder": args.eeg_folder,
                "img_folder": args.img_folder,
                'project': args.project,
                "model_type": args.model,
                "subject_id": args.subject_id.split(','),
                "time_window": [args.start_time, args.end_time],
                "channels": args.channels.split(','),
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "image_size": image_size,
                "early_stopping_patience": args.early_stopping,
                "seed": args.seed,
                "loss": args.loss,
                'average_eeg': args.average_eeg,
                "latent_mapping": args.latent_mapping,
                "device": device,
                "save_model": args.save_model
            }
    
    EEG_dir = config['eeg_folder']
    img_dir = config['img_folder']
    img_metadata = np.load(os.path.join(img_dir, 'image_metadata.npy'), allow_pickle=True).item()
    train_d, validation_d,ntimes = load_multiple_subjects(subject_ids=config['subject_id'], eeg_dir=EEG_dir, img_dir=img_dir, 
                                                    img_metadata=img_metadata, start_time=config['time_window'][0], 
                                                    end_time=config['time_window'][1],desired_channels=config['channels'],
                                                    image_size=(config['image_size'][1],config['image_size'][2]),compressor=config['latent_mapping'],training=True,average=config['average_eeg'])

    train_dataloader = DataLoader(train_d, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_d, batch_size=config['batch_size'], shuffle=False)
    # combined_dataset = ConcatDataset([train_d, validation_d])
    # train_dataloader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True)
    # Set the random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # print(config['channels'])
    if config['channels'][0] == 'All':
        n_channels = len(train_d[0][1])
    else:
        n_channels = len(config['channels'])
    if config['model_type']== 'encoder_low_level':
        eeg_model = encoder_low_level(num_channels=n_channels, sequence_length=ntimes).to(device)
    elif config['model_type'] == 'EEGConformer':
        eeg_model = EEGConformer_Deconv(n_outputs=2, n_chans=n_channels, n_filters_time=90, 
                                    filter_time_length=20, pool_time_length=5, pool_time_stride=5, 
                                    drop_prob=0.5, att_depth=3, att_heads=30, att_drop_prob=0.5, 
                                    final_fc_length='auto', return_features=False, n_times=ntimes, 
                                    chs_info=None, input_window_seconds=None, sfreq=None, n_classes=None, 
                                    n_channels=None, input_window_samples=None).to(device)
    elif config['model_type'] == 'encoder_low_level_channelwise':
        eeg_model = encoder_low_level(num_channels=n_channels, sequence_length=ntimes).to(device)
    elif config['model_type'] == 'ATMS':
        ATM_config = Config(seq_len=ntimes,ATMoutput=1024)
        eeg_model=ATMS_DM(ATM_config)
    # elif config['model_type'] == 'ATMS_Res_attention':
    #     ATM_config = Config(seq_len=ntimes,ATMoutput=1024)
    #     eeg_model=ATMS_Res_attention(ATM_config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    print(f"Model type: {config['model_type']}")
        

    # def init_wandb(project_name=config['project'], run_name=None,config=config):
    #     if run_name is None:
    #         run_name = f"{config['subject_id'][0]}-{config['model_type']}-C{n_channels}-{config['time_window'][1]}s-avg{config['average_eeg']}"
    #     wandb.init(project=project_name, 
    #             name=run_name, 
    #             config=config)
        
    #     # Log model architecture as text
    #     wandb.run.summary["model_architecture"] = config['model_type']
        
    #     return wandb.run

    # exit()
    # run = init_wandb(config=config)

    main_train_loop(eeg_model, train_dataloader, validation_dataloader,device,config,
                    model_type=config['model_type'],save_model=config['save_model'])
    # main_train_loop(eeg_model, train_dataloader, device, 
    #                  config,model_type='EEGconformer',save_model=False)
if __name__ == '__main__':
        # Add this before your training loop
    # print(f"Config: {config['average_eeg']}")
    # gpu0_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
    # gpu1_memory = torch.cuda.memory_allocated(1) / (1024 ** 3)
    # print(f"GPU 0 memory usage: {gpu0_memory:.2f} GB")
    # print(f"GPU 1 memory usage: {gpu1_memory:.2f} GB")
    main()
    torch.cuda.empty_cache()
