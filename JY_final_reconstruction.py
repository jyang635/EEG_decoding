import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import re
import argparse
import csv


from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from diffusion_prior import DiffusionPriorUNet, Pipe,EmbeddingDataset
from custom_pipeline_low_level import Generator4Embeds

from ATMS_reconstruction import ATMS as ATMS_highlevel
from JY_low_level_encoders import Deconv_EEGConformer,encoder_low_level,encoder_low_level_channelwise,ATMS_Deconv,Config
from JY_ThingsData import load_multiple_subjects
from JY_Image_metrics import compute_metrics,save_model_results_to_csv
from JY_VAE_compare import VAE_reconstruction

from torchvision import transforms
# from PIL import Image
import torchvision.transforms as T


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
def extract_highlevel_embeddings(eegmodel, dataloader, device, subject_id=None):
    eegmodel.eval()
    eeg_embeddings=[]
    image_list=[]

    with torch.no_grad():
        for batch_idx, (img, eeg_data) in enumerate(dataloader):            
            eeg_data = eeg_data.to(device)
            subject_ids = extract_id_from_string(subject_id)
            batch_size =eeg_data.size(0)
            subject_ids = torch.full((batch_size,), subject_ids, dtype=torch.long).to(device)
            eeg_features = eegmodel(eeg_data,subject_ids).float()
            eeg_embeddings.append(eeg_features.cpu())
            image_list.append(img.cpu())
    
    eeg_embeddings = torch.cat(eeg_embeddings, dim=0)
    image_list = torch.cat(image_list, dim=0)
    torch.cuda.empty_cache()
    return image_list,eeg_embeddings

def FinalSDXL_reconstruction(highlevel_feature,lowlevel_image,DM_prior,device,image_size=(256, 256),guidance_scale=50.0):
    recon_list=[]
    with torch.no_grad():
        for i in range(highlevel_feature.shape[0]):
            # if i==3:
            #     print("Stopping at 3")
            #     break
            ref_img = lowlevel_image[i].unsqueeze(0).to(device)
            highlevel_ref = highlevel_feature[i].unsqueeze(0).to(device)
            generator = Generator4Embeds(num_inference_steps=10, device=device, img2img_strength=0.85, 
                                        low_level_image=ref_img, low_level_latent=None) 
            h = DM_prior.generate(c_embeds=highlevel_ref, num_inference_steps=10, guidance_scale=guidance_scale)     
            eeg_gen = generator.generate(h,generator=None)
            eeg_gen=transforms.ToTensor()(eeg_gen).cpu()
            # eeg_gen=transforms.Resize(image_size)(eeg_gen)
            recon_list.append(eeg_gen)
            del generator,h
            torch.cuda.empty_cache()
    
    recon_list = torch.cat(recon_list, dim=0)
    recon_list=transforms.Resize(image_size)(recon_list)
    recon_list = recon_list.view(-1, 3, 256, 256)
    print( f"Recon_list shape: {recon_list.shape}")
    torch.cuda.empty_cache()
    return recon_list



    
    # print(f"Results for model '{config['model_type']}' saved to {csv_file_path}")
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Conformer+deconv')
    
    # Add your command-line arguments
    parser.add_argument('--eeg_folder', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz')
    parser.add_argument('--img_folder', type=str, default='/home/yjk122/IP_temp/ThingsEEG/image')
    parser.add_argument('--model_path', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Generation/models',help='Path to the pre-trained model')
    parser.add_argument('--subject_id', type=str, default='sub-01', help='Subject ID to analyze')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time for analysis window')
    parser.add_argument('--end_time', type=float, default=1.0, help='End time for analysis window')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--model', type=str,choices=['encoder_low_level', 'encoder_low_level_channelwise', 'EEGConformer','ATMS'], default='ATMS')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--channels', type=str, default='All', 
                        help='EEG channels to use (comma-separated)')
    parser.add_argument('--image_size', type=str, default="256,256", help='size of the image'),
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--average_eeg', action='store_true', help='Whether to average EEG data')
    
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    image_size = args.image_size.split(',')
    image_size = (3,int(image_size[0]), int(image_size[1]))
    config = {
                "eeg_folder": args.eeg_folder,
                "img_folder": args.img_folder,
                "model_type": args.model,
                "model_path": args.model_path,
                "batch_size": args.batch_size,
                "subject_id": args.subject_id.split(','),
                "time_window": [args.start_time, args.end_time],
                "channels": args.channels.split(','),
                "image_size": image_size,
                "seed": args.seed,
                'average_eeg': args.average_eeg,
                "device": device
            }
    # Load the EEG data and the image data
    EEG_dir = config['eeg_folder']
    img_dir = config['img_folder']
    img_metadata = np.load(os.path.join(img_dir, 'image_metadata.npy'), allow_pickle=True).item()
    test_d,ntimes = load_multiple_subjects(subject_ids=config['subject_id'], eeg_dir=EEG_dir, img_dir=img_dir, 
                                                    img_metadata=img_metadata, start_time=config['time_window'][0], 
                                                    end_time=config['time_window'][1],desired_channels=config['channels'],
                                                    image_size=(config['image_size'][1],config['image_size'][2]),compressor=None,training=False,average=config['average_eeg'])
    test_loader = DataLoader(test_d, batch_size=config['batch_size'], shuffle=False)

    # Extract the high-level features from the eeg
    if config['channels'][0] == 'All':
        n_channels = len(test_d[0][1])
    else:
        n_channels = len(config['channels'])
    eeg_highlevel=ATMS_highlevel(63,250)
    #get the number of parameters in the model
    num_params = sum(p.numel() for p in eeg_highlevel.parameters() if p.requires_grad)
    # m_path=config['model_path']
    path_modelstate=f"{config['model_path']}/contrast/{config['model_type']}/{config['subject_id'][0]}"
    only_folders = [f for f in os.listdir(path_modelstate) if os.path.isdir(os.path.join(path_modelstate, f))]
    checkpoint= torch.load(os.path.join(path_modelstate, only_folders[0],'40.pth'), map_location=device,weights_only=True)
    # Load the model state
    eeg_highlevel.load_state_dict(checkpoint)
    eeg_highlevel.to(device)
    eeg_highlevel.eval()
    #Get the high-level features
    image_list, highlevel_embeddings = extract_highlevel_embeddings(eeg_highlevel, test_loader, device, subject_id=config['subject_id'][0])
    print(f"High-level embeddings obtained for {config['subject_id'][0]}")
    del eeg_highlevel
    torch.cuda.empty_cache()

    #Obtain the low-level images
    # Load the low-level model
    if config['model_type'] == 'encoder_low_level':
        low_level_model = encoder_low_level()
    elif config['model_type'] == 'encoder_low_level_channelwise':
        low_level_model = encoder_low_level_channelwise()
    elif config['model_type'] == 'EEGConformer':
        low_level_model = Deconv_EEGConformer()
    elif config['model_type'] == 'ATMS':
        ATM_config = Config(seq_len=ntimes,ATMoutput=1024)
        low_level_model=ATMS_Deconv(ATM_config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    low_level_model.to(device)
    low_level_model.eval()
    # Load the VAE model
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")
    if hasattr(pipe, 'vae'):
        for param in pipe.vae.parameters():
            param.requires_grad = False
    vae = pipe.vae.to(device)
    # vae = pipe.vae
    del pipe
    vae.requires_grad_(False)
    vae.eval()
    print(f"VAE loaded")
    _,lowlevel_recon_list = VAE_reconstruction(low_level_model, test_loader,vae, device,image_size=(512,512),model_type=config['model_type'],subject_id=config['subject_id'][0])
    print(f"Low-level reconstructions obtained for {config['model_type']} {config['subject_id'][0]}")
    del vae, low_level_model
    torch.cuda.empty_cache()

    # Load the diffusion prior
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(
        torch.load(f"{config['model_path']}/DM_highlevel/{config['subject_id'][0]}/highlevel_DM.pth", map_location=device))
    print(f"DM prior loaded")
    #### Initialize the Image generator
    # 

    # Get the GT images and the eeg embeddings
    Final_recon_list = FinalSDXL_reconstruction(highlevel_embeddings,lowlevel_recon_list,pipe,device='cuda:1',image_size=(256, 256),guidance_scale=50.0)
    print(f"Final reconstructions obtained for {config['model_type']} {config['subject_id'][0]}")


    # Create a directory for the reconstructed images if it doesn't exist
    recon_dir = os.path.join(config['model_path'], f"Final_reconstructions/{config['model_type']}_{config['subject_id'][0]}")
    os.makedirs(recon_dir, exist_ok=True)
    # Convert tensor images to PIL images and save them
    to_pil = T.ToPILImage()
    for i in range(min(10, Final_recon_list.shape[0])):
        img = to_pil(Final_recon_list[i])
        img.save(os.path.join(recon_dir, f"recon_{i}.png"))

    # Also save the original images for comparison
    for i in range(min(10, image_list.shape[0])):
        img = to_pil(image_list[i])
        img.save(os.path.join(recon_dir, f"original_{i}.png"))

    print(f"Saved first 10 reconstructions to {recon_dir}")


    # Compute the metrics
    metrics_results = compute_metrics(image_list, Final_recon_list,device)

    # Save the metrics to a CSV file
    csv_file_path =os.path.join(config['model_path'], "Final_model_results.csv")
    # save the first 5 reconstructions as PIL to model_path
    # Save the first 5 reconstructions as PIL images

    # Check if the file exists
    if not os.path.exists(csv_file_path):
        # Define the header for the CSV file
        header = [
            "model_name",
            "subject_id",
            "channels",
            "start_time",
            "end_time",
            "number_of_images",
            "number_of_parameters",
            "Image_size",
            "mse",
            "lpips",
            "pixel_corr",
            "ssim",
            "Alex_2",
            "Alex_5",
            "Inception",
            "CLIP",
            "SwAV"
        ]

        # Create the CSV file and write the header
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

        print(f"CSV file '{csv_file_path}' created successfully.")
    else:
        print(f"CSV file '{csv_file_path}' already exists.")


    save_model_results_to_csv(metrics_results, csv_file_path, config,num_params,num_images=len(test_d))
    print(f"Results for {config['model_type']} saved to {csv_file_path}")
if __name__ == '__main__':
    main()
