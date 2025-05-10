import os
import torch
from torch.utils.data import DataLoader

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
import re
import argparse
import csv
from torch import Tensor
import math

# from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from diffusion_prior import DiffusionPriorUNet, Pipe,EmbeddingDataset
from custom_pipeline import Generator4Embeds

from ATMS_reconstruction import ATMS
from JY_ThingsData import load_multiple_subjects
from JY_Image_metrics import compute_metrics,save_model_results_to_csv


from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import torchvision.transforms as T

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
def SDXL_reconstruction(eegmodel, dataloader,SDXL,device,image_size=(256, 256),subject_id=None,guidance_scale=5.0):
    eegmodel.eval()
    recon_list=[]
    image_list=[]

    with torch.no_grad():
        for batch_idx, (img, eeg_data) in enumerate(dataloader): 
            # if batch_idx ==3:
            #     break      
            eeg_data = eeg_data.to(device)
            subject_ids = extract_id_from_string(subject_id)
            batch_size =eeg_data.size(0)
            subject_ids = torch.full((batch_size,), subject_ids, dtype=torch.long).to(device)
            eeg_features = eegmodel(eeg_data,subject_ids).float()
            eeg_gen = SDXL.generate(eeg_features.to(dtype=torch.float16))
            eeg_gen=transforms.ToTensor()(eeg_gen)
            # eeg_gen=transforms.Resize(image_size)(eeg_gen)
            recon_list.append(eeg_gen.cpu())
            image_list.append(img.cpu())
            # torch.cuda.empty_cache()
    
    recon_list = torch.cat(recon_list, dim=0)
    #resize recon_list to the same size as image_list
    # recon_list=transforms.Resize((img.shape[2], img.shape[3]))(recon_list)
    # recon_list= (recon_list+1)/2
    recon_list=transforms.Resize(image_size)(recon_list)
    recon_list = recon_list.view(-1, 3, 256, 256)
    print( f"Recon_list shape: {recon_list.shape}")
    
    ### make sure the recon_list is in the range of 0-1
    image_list = torch.cat(image_list, dim=0)
    image_list=transforms.Resize(image_size)(image_list)
    image_list=image_list.clamp(0,1)
    # recon_list=positive_images(recon_list)
    return image_list,recon_list



    
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
    # Load the pre-trained model
    if config['channels'][0] == 'All':
        n_channels = len(test_d[0][1])
    else:
        n_channels = len(config['channels'])
    eeg_model=ATMS(63,250)
    #get the number of parameters in the model
    num_params = sum(p.numel() for p in eeg_model.parameters() if p.requires_grad)
    # m_path=config['model_path']
    path_modelstate=f"{config['model_path']}/contrast/{config['model_type']}/{config['subject_id'][0]}"
    only_folders = [f for f in os.listdir(path_modelstate) if os.path.isdir(os.path.join(path_modelstate, f))]
    checkpoint= torch.load(os.path.join(path_modelstate, only_folders[0],'40.pth'), map_location=device,weights_only=True)
    # model_path = os.path.join(path_modelstate, matching_files)
    
    # Load the model state
    # checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    eeg_model.load_state_dict(checkpoint)
    eeg_model.to(device)
    eeg_model.eval()
    test_loader = DataLoader(test_d, batch_size=1, shuffle=False)


    generator = Generator4Embeds(num_inference_steps=4, device=device)
    # torch.cuda.empty_cache()

    # Get the GT images and the eeg embeddings
    image_list,recon_list = SDXL_reconstruction(eeg_model, 
                                                test_loader,SDXL=generator,
                                                device=device,subject_id=config['subject_id'][0],
                                                guidance_scale=50.0)
    print(f"Reconstructions obtained for {config['subject_id'][0]}")

        # Create a directory for the reconstructed images if it doesn't exist
    recon_dir = os.path.join(config['model_path'], f"noDMHighlevel_reconstructions/{config['model_type']}_{config['subject_id'][0]}")
    os.makedirs(recon_dir, exist_ok=True)
    # Convert tensor images to PIL images and save them
    to_pil = T.ToPILImage()
    for i in range(min(30, recon_list.shape[0])):
        img = to_pil(recon_list[i])
        img.save(os.path.join(recon_dir, f"recon_{i}.png"))

    # Also save the original images for comparison
    # for i in range(min(10, image_list.shape[0])):
    #     img = to_pil(image_list[i])
    #     img.save(os.path.join(recon_dir, f"original_{i}.png"))

    print(f"Saved first 10 reconstructions to {recon_dir}")

    metrics_results = compute_metrics(image_list, recon_list,device)

    # Save the metrics to a CSV file
    csv_file_path =os.path.join(config['model_path'], "noDMHighlevel_model_results.csv")
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
