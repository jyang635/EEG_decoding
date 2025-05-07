import os
import torch
from torch.utils.data import DataLoader

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
import argparse
import csv
from torch import Tensor
import math

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *

from JY_low_level_encoders import Deconv_EEGConformer,encoder_low_level,encoder_low_level_channelwise,ATMS_Deconv,Config
from JY_ThingsData import load_multiple_subjects


from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
from diffusers.image_processor import VaeImageProcessor
image_processor = VaeImageProcessor()

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
def getEEGembed(eegmodel, dataloader, vae,device,image_size=(256, 256)):
    eegmodel.eval()
    recon_list=[]
    image_list=[]

    with torch.no_grad():
        for batch_idx, (img, eeg_data) in enumerate(dataloader):            
            eeg_data = eeg_data.to(device)
            eeg_features = eegmodel(eeg_data).float()
            image_list.append(img)
            # z= eeg_features.to('cuda:1')
            z= eeg_features.to(device)
            x_rec = vae.decode(z).sample
            recon_list.append(x_rec.cpu())
            del z,x_rec
            # torch.cuda.empty_cache()
    
    recon_list = torch.cat(recon_list, dim=0)
    #resize recon_list to the same size as image_list
    # recon_list=transforms.Resize((img.shape[2], img.shape[3]))(recon_list)
    recon_list= (recon_list+1)/2
    recon_list=transforms.Resize(image_size)(recon_list)
    
    ### make sure the recon_list is in the range of 0-1
    image_list = torch.cat(image_list, dim=0)
    image_list=transforms.Resize(image_size)(image_list)
    image_list=image_list.clamp(0,1)
    # recon_list=positive_images(recon_list)
    return image_list,recon_list
def compute_metrics(test_images, test_recon,device):
    # Initialize the metrics
    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device)

    # Dictionary to store results
    metrics_results = {
        "mse": [],
        "ssim": [],
        "lpips": [],
        "pixel_corr": []
    }

    # Compute metrics for each pair of ground truth and reconstruction
    for i in range(test_images.shape[0]):
        gt = test_images[i].unsqueeze(0).to(device)
        recon = test_recon[i].unsqueeze(0).to(device)

        # Compute MSE
        mse_value = mse(recon, gt).item()

        # Compute SSIM
        ssim_value = ssim(recon, gt).item()

        # Compute LPIPS
        lpips_value = lpips(recon, gt).item()

        # Compute pixel-wise correlation
        gt_flat = gt.cpu().numpy().flatten()
        recon_flat = recon.cpu().numpy().flatten()
        pixel_corr_value = np.corrcoef(gt_flat, recon_flat)[0, 1]

        # Append results to the dictionary
        metrics_results["mse"].append(mse_value)
        metrics_results["ssim"].append(ssim_value)
        metrics_results["lpips"].append(lpips_value)
        metrics_results["pixel_corr"].append(pixel_corr_value)

    return metrics_results

def save_model_results_to_csv(metrics_stats_dict, csv_file_path, config,num_params,num_images=200):
    """
    Save model evaluation results to CSV file.
    If the model already exists in the CSV, its row will be updated.
    
    Args:
        model_name (str): Name of the model
        metrics_stats_dict (dict): Dictionary containing metrics statistics
        csv_file_path (str): Path to the CSV file
        config (dict): Configuration dictionary containing model parameters
        num_params (int): Number of parameters in the model
        num_images (int): Number of images used for evaluation
    """
    # Extract metrics for the specified model
    model_metrics = metrics_stats_dict[config['model_type']]
    
    # Prepare data for CSV
    row_data = {
        "model_name": config['model_type'],
        "subject_id": config['subjects'][0],  # Assuming this is always the first subject based on context
        "channels": config['channels'],  # Assuming this is always "All" based on context
        "start_time": config['time_window'][0],   # Could be filled with actual data if available
        "end_time": config['time_window'][1],     # Could be filled with actual data if available
        "number_of_images": num_images,  # Number of images used
        "number_of_parameters": num_params,  # Could be filled with actual model parameters count
        "Image_size": (config['image_size'][1],config['image_size'][2]),  # Assuming this is always 512 based on context
        "test_mse_mu": model_metrics["mse_mu"],
        "test_mse_std": model_metrics["mse_std"],
        "test_mse_sem": model_metrics["mse_sem"],
        "test_PixCorr_mu": model_metrics["pixel_corr_mu"],
        "test_PixCorr_std": model_metrics["pixel_corr_std"],
        "test_PixCorr_sem": model_metrics["pixel_corr_sem"],
        "test_ssim_mu": model_metrics["ssim_mu"],
        "test_ssim_std": model_metrics["ssim_std"],
        "test_ssim_sem": model_metrics["ssim_sem"],
        "test_lpips_mu": model_metrics["lpips_mu"],
        "test_lpips_std": model_metrics["lpips_std"],
        "test_lpips_sem": model_metrics["lpips_sem"],
    }
    
    # Read existing CSV data
    existing_data = []
    with open(csv_file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        for row in reader:
            if row["model_name"] != config['model_type'] & row["subject_id"]!=config['subjects'][0]:  # Keep all rows except the one with matching model name
                existing_data.append(row)
    
    # Write back all data including the new/updated row
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        
        # Write existing rows (excluding the model that's being updated)
        for row in existing_data:
            writer.writerow(row)
        
        # Write the new/updated row
        writer.writerow(row_data)
    
    print(f"Results for model '{config['model_type']}' saved to {csv_file_path}")
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Conformer+deconv')
    
    # Add your command-line arguments
    parser.add_argument('--eeg_folder', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz')
    parser.add_argument('--img_folder', type=str, default='/home/yjk122/IP_temp/ThingsEEG/image')
    parser.add_argument('--model_path', type=str, default='/home/yjk122/IP_temp/EEG_Image_decode/Generation/models/low_level',help='Path to the pre-trained model')
    parser.add_argument('--subject_id', type=str, default='sub-01', help='Subject ID to analyze')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time for analysis window')
    parser.add_argument('--end_time', type=float, default=0.5, help='End time for analysis window')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--model', type=str,choices=['encoder_low_level', 'encoder_low_level_channelwise', 'EEGConformer','ATMS'], default='encoder_low_level')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--channels', type=str, default='O1,Oz,O2', 
                        help='EEG channels to use (comma-separated)')
    parser.add_argument('--image_size', type=str, default="256,256", help='size of the image'),
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--average_eeg', action='store_true', help='Whether to average EEG data')
    parser.add_argument('--latent_mapping', default=None, help='Whether to use latent mapping')
    
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
                "batch_size": args.batch_size,  
                "image_size": image_size,
                "seed": args.seed,
                'average_eeg': args.average_eeg,
                "latent_mapping": args.latent_mapping,
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
    #get the number of parameters in the model
    num_params = sum(p.numel() for p in eeg_model.parameters() if p.requires_grad)
    # m_path=config['model_path']
    path_modelstate=f"{config['model_path']}/{config['model_type']}/{config['subject_id'][0]}/C{n_channels}-{config['time_window'][1]}s-avg{config['average_eeg']}"
    matching_files = [f for f in os.listdir(path_modelstate) if f.startswith('model')][0]
    model_path = os.path.join(path_modelstate, matching_files)
    
    # Load the model state
    checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    eeg_model.load_state_dict(checkpoint)
    eeg_model.to(device)
    eeg_model.eval()


    test_loader = DataLoader(test_d, batch_size=config['batch_size'], shuffle=False)
    # Load the VAE model
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")
    if hasattr(pipe, 'vae'):
        for param in pipe.vae.parameters():
            param.requires_grad = False
    # vae = pipe.vae.to('cuda:1')
    vae = pipe.vae
    del pipe
    vae.requires_grad_(False)
    vae.eval()
    print(f"VAE loaded")

    # Get the GT images and the eeg embeddings
    image_list,recon_list = getEEGembed(eeg_model, test_loader,vae, device)
    print(f"Reconstructions obtained for {config['model_type']} {config['subject_id'][0]}")

    metrics_models_dict= {config['model_type']: 0}
    metrics_stats_dict = {config['model_type']: 0 }


    metrics_results = compute_metrics(image_list, recon_list,device)
    metrics_models_dict[config['model_type']] = metrics_results
    metrics_stats_dict[config['model_type']] = {
        "mse_mu": round(np.mean(metrics_results["mse"]), 4),
        "mse_std": round(np.std(metrics_results["mse"]), 4),
        "mse_sem": round(np.std(metrics_results["mse"]) / math.sqrt(len(metrics_results["mse"])), 4),
        "ssim_mu": round(np.mean(metrics_results["ssim"]), 4),
        "ssim_std": round(np.std(metrics_results["ssim"]), 4),
        "ssim_sem": round(np.std(metrics_results["ssim"]) / math.sqrt(len(metrics_results["ssim"])), 4),
        "lpips_mu": round(np.mean(metrics_results["lpips"]), 4),
        "lpips_std": round(np.std(metrics_results["lpips"]), 4),
        "lpips_sem": round(np.std(metrics_results["lpips"]) / math.sqrt(len(metrics_results["lpips"])), 4),
        "pixel_corr_mu": round(np.mean(metrics_results["pixel_corr"]), 4),
        "pixel_corr_std": round(np.std(metrics_results["pixel_corr"]), 4),
        "pixel_corr_sem": round(np.std(metrics_results["pixel_corr"]) / math.sqrt(len(metrics_results["pixel_corr"])), 4)}
    print(f"Metrics computed for {config['model_type']} {config['subject_id'][0]}")
    # Save the metrics to a CSV file
    csv_file_path =os.path.join(config['model_path'], "lowlevel_model_results.csv")
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
            "test_mse_mu",
            "test_mse_std",
            "test_mse_sem",
            "test_PixCorr_mu",
            "test_PixCorr_std",
            "test_PixCorr_sem",
            "test_ssim_mu",
            "test_ssim_std",
            "test_ssim_sem",
            "test_lpips_mu",
            "test_lpips_std",
            "test_lpips_sem",
        ]

        # Create the CSV file and write the header
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()

        print(f"CSV file '{csv_file_path}' created successfully.")
    else:
        print(f"CSV file '{csv_file_path}' already exists.")



    save_model_results_to_csv(config['model_type'], metrics_stats_dict, csv_file_path, config,num_params,num_images=len(test_d))
    print(f"Results for {config['model_type']} saved to {csv_file_path}")
if __name__ == '__main__':
    main()
