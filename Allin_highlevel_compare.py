import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import re
import argparse
import csv


from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *

from EEG_low_level_encoders import Config
from EEG_ThingsData import load_multiple_subjects
from EEG_Image_metrics import compute_metrics,save_model_results_to_csv
from ALLin_highlevel import ATMS_DM
from custom_pipeline import Generator4Embeds


# from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import torchvision.transforms as T

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
def geteeg_features(eegmodel, dataloader,device,image_size=(256, 256),model_type='ATMS_Deconv',subject_id=None,guidance_scale=50.0):
    eegmodel.eval()
    eeg_featurelist=[]
    image_list=[]
    print(f'Guidance scale: {guidance_scale}')
    # print(f'Shape of EEG data: {dataloader.dataset[0][1].shape}')
    with torch.no_grad():
        for batch_idx, (img, eeg_data) in enumerate(dataloader):            
            eeg_data = eeg_data.to(device)

            subject_ids = extract_id_from_string(subject_id)
            batch_size =eeg_data.size(0)
            subject_ids = torch.full((batch_size,), subject_ids, dtype=torch.long).to(device)
            eeg_features = eegmodel.generate(eeg_data,subject_ids,num_inference_steps=50, guidance_scale=guidance_scale)

            image_list.append(img)
            # z= eeg_features.to('cuda:1')
            eeg_featurelist.append(eeg_features.cpu())
            print(f"Batch {batch_idx+1}/{len(dataloader)} processed")


            torch.cuda.empty_cache()

    
    eeg_featurelist = torch.cat(eeg_featurelist, dim=0)

    
    ### make sure the recon_list is in the range of 0-1
    image_list = torch.cat(image_list, dim=0)
    image_list=transforms.Resize(image_size)(image_list)
    image_list=image_list.clamp(0,1)
    # recon_list=positive_images(recon_list)
    return image_list,eeg_featurelist

def SDXL_reconstruction(eeg_featurelist,SDXL,device,image_size=(256, 256)):
    recon_list=[]


    with torch.no_grad():
        for i in range(eeg_featurelist.shape[0]):
            eeg_gen = SDXL.generate(eeg_featurelist[i:i+1].to(dtype=torch.float16).to(device))
            eeg_gen=transforms.ToTensor()(eeg_gen)
            # eeg_gen=transforms.Resize(image_size)(eeg_gen)
            recon_list.append(eeg_gen.cpu())
            print(f"Batch {i}/{eeg_featurelist.shape[0]} processed")
            torch.cuda.empty_cache()

            
    
    recon_list = torch.cat(recon_list, dim=0)
    #resize recon_list to the same size as image_list
    # recon_list=transforms.Resize((img.shape[2], img.shape[3]))(recon_list)
    # recon_list= (recon_list+1)/2
    recon_list=transforms.Resize(image_size)(recon_list)
    recon_list = recon_list.view(-1, 3, 256, 256)
    print( f"Recon_list shape: {recon_list.shape}")

    return recon_list

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
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
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
                "batch_size": args.batch_size,  
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

    if config['model_type'] == 'ATMS':
        ATM_config = Config(seq_len=ntimes,ATMoutput=1024)
        eeg_model=ATMS_DM(ATM_config)
    # elif config['model_type'] == 'ATMS_Res_attention':
    #     ATM_config = Config(seq_len=ntimes,ATMoutput=1024)
    #     eeg_model=ATMS_Res_attention(ATM_config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    #get the number of parameters in the model
    num_params = sum(p.numel() for p in eeg_model.parameters() if p.requires_grad)
    # m_path=config['model_path']
    
    # Load the model state
    checkpoint = torch.load(f"{config['model_path']}/Highlevel_Allin/{config['model_type']}/{config['subject_id'][0]}/Highlevel_DM.pth", map_location=device)
    eeg_model.load_state_dict(checkpoint)
    eeg_model.to(device)
    eeg_model.eval()


    test_loader = DataLoader(test_d, batch_size=config['batch_size'], shuffle=False)
    # Get the GT images and the eeg embeddings
    image_list,eeg_featurelist = geteeg_features(eeg_model, test_loader, device,model_type=config['model_type'],subject_id=config['subject_id'][0],guidance_scale=50)
    del eeg_model,checkpoint
    torch.cuda.empty_cache()
    print(f"EEG features extracted for {config['model_type']} {config['subject_id'][0]}")



    #### Initialize the Image generator
    generator = Generator4Embeds(num_inference_steps=4, device=device)
    torch.cuda.empty_cache()
    print(f"Reconstructions obtained for {config['model_type']} {config['subject_id'][0]}")

    recon_list = SDXL_reconstruction(eeg_featurelist, generator, device, image_size=(image_size[1], image_size[2]))

    # metrics_models_dict= {config['model_type']: 0}
    # metrics_stats_dict = {config['model_type']: 0 }

    # Create a directory for the reconstructed images if it doesn't exist
    recon_dir = os.path.join(config['model_path'], f"ALLin_Highlevel_reconstructions/DM_{config['model_type']}_{config['subject_id'][0]}_avg{config['average_eeg']}")
    os.makedirs(recon_dir, exist_ok=True)
    # Convert tensor images to PIL images and save them
    to_pil = T.ToPILImage()
    for i in range(min(30, recon_list.shape[0])):
        img = to_pil(recon_list[i])
        img.save(os.path.join(recon_dir, f"recon_{i}.png"))

    # # Also save the original images for comparison
    # for i in range(min(30, image_list.shape[0])):
    #     img = to_pil(image_list[i])
    #     img.save(os.path.join(recon_dir, f"original_{i}.png"))

    print(f"Saved first 30 reconstructions to {recon_dir}")

    metrics_results = compute_metrics(image_list, recon_list,device)

    # Save the metrics to a CSV file
    csv_file_path =os.path.join(config['model_path'], "ALLin_Highlevel_model_results.csv")
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
