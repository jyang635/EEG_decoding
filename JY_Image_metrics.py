
import numpy as np

import scipy as sp
import pandas as pd
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision.models import alexnet, AlexNet_Weights
import clip
import csv
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
# print("device:",device)

import utils
seed=42
# utils.seed_everything(seed=seed)

from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, device,feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1

def Pixcorr(test_recon, test_images,device):
    # Compute metrics for each pair of ground truth and reconstruction
    metrics_results = []
    for i in range(test_images.shape[0]):
        gt = test_images[i].unsqueeze(0).to(device)
        recon = test_recon[i].unsqueeze(0).to(device)

        # Compute pixel-wise correlation
        gt_flat = gt.cpu().numpy().flatten()
        recon_flat = recon.cpu().numpy().flatten()
        pixel_corr_value = np.corrcoef(gt_flat, recon_flat)[0, 1]
        metrics_results.append(pixel_corr_value)
        # metrics_results = np.array(metrics_results)
    return metrics_results
       
def AlexNet_2way(test_recon, test_images,device):
    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    Layer2_all_per_correct = two_way_identification(test_recon.to(device).float(), test_images, 
                                                          alex_model, preprocess, device,'features.4')
    # alexnet2 = np.mean(all_per_correct)
    Lyaer_5_all_per_correct = two_way_identification(test_recon.to(device).float(), test_images, 
                                                          alex_model, preprocess, device,'features.11')
    # alexnet5 = np.mean(all_per_correct)
    return Layer2_all_per_correct, Lyaer_5_all_per_correct

def Inception_2way(test_recon, test_images,device):
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(test_recon, test_images,
                                            inception_model, preprocess, device,'avgpool')
    return all_per_correct

def CLIP_2way(test_recon, test_images,device):
    # Load the CLIP model
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(test_recon, test_images,
                                            clip_model.encode_image, preprocess,device) # final layer
    return all_per_correct

def EfficientNet_metric(test_recon, test_images,device):
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool']).to(device)
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(test_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(test_recon))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    return effnet

def Swav_metric(test_recon, test_images,device):
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool']).to(device)
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(test_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(test_recon))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    return swav

def compute_metrics(test_images, test_recon,device):
    # Initialize the metrics
    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to(device)
    test_images = test_images.to(device)
    test_recon = test_recon.to(device)
    # Dictionary to store results
    metrics_results = {
        "mse": [],
        "ssim": [],
        "lpips": [],
        "pixel_corr": [],
        "Alex_2": [],
        "Alex_5": [],
        "Inception": [],
        "CLIP": [],
        "SwAV": []

    }
    Alex2,Alex5=AlexNet_2way(test_recon, test_images,device=device)
    Inception_metric=Inception_2way(test_recon, test_images,device=device)
    CLIP_metric=CLIP_2way(test_recon, test_images,device=device)
    SwAV_metric=Swav_metric(test_recon, test_images,device=device)
    pixel_corr = Pixcorr(test_recon, test_images,device=device)
    metrics_results["pixel_corr"]=np.mean(pixel_corr)
    metrics_results["Alex_2"]=Alex2
    metrics_results["Alex_5"]=Alex5
    metrics_results["Inception"]=Inception_metric
    metrics_results["CLIP"]=CLIP_metric
    metrics_results["SwAV"]=SwAV_metric
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
        

        # Append results to the dictionary
        metrics_results["mse"].append(mse_value)
        metrics_results["ssim"].append(ssim_value)
        metrics_results["lpips"].append(lpips_value)
    metrics_results["mse"]= np.array(metrics_results["mse"]).mean()
    metrics_results["ssim"]= np.array(metrics_results["ssim"]).mean()
    metrics_results["lpips"]= np.array(metrics_results["lpips"]).mean()

    return metrics_results

def save_model_results_to_csv(metrics_results, csv_file_path, config,num_params,num_images=200):
    """
    Save model evaluation results to CSV file.
    If the model and subject already exist in the CSV, its row will be updated.
    
    Args:
        model_name (str): Name of the model
        metrics_stats_dict (dict): Dictionary containing metrics statistics
        csv_file_path (str): Path to the CSV file
        config (dict): Configuration dictionary containing model parameters
        num_params (int): Number of parameters in the model
        num_images (int): Number of images used for evaluation
    """
    # Extract metrics for the specified model
    # model_metrics = metrics_results[config['model_type']]
    
    # Prepare data for CSV
    row_data = {
        "model_name": config['model_type'],
        "subject_id": config['subject_id'][0],  # Assuming this is always the first subject based on context
        "channels": config['channels'],  # Assuming this is always "All" based on context
        "start_time": config['time_window'][0],   # Could be filled with actual data if available
        "end_time": config['time_window'][1],     # Could be filled with actual data if available
        "number_of_images": num_images,  # Number of images used
        "number_of_parameters": num_params,  # Could be filled with actual model parameters count
        "Image_size": (config['image_size'][1],config['image_size'][2]),  # Assuming this is always 512 based on context
        "mse": metrics_results["mse"],
        "lpips": metrics_results["lpips"],
        "pixel_corr": metrics_results["pixel_corr"],
        "ssim": metrics_results["ssim"],
        "Alex_2": metrics_results["Alex_2"],
        "Alex_5": metrics_results["Alex_5"],
        "Inception": metrics_results["Inception"],
        "CLIP": metrics_results["CLIP"],
        "SwAV": metrics_results["SwAV"]
        # "test_mse_mu": model_metrics["mse_mu"],
        # "test_mse_std": model_metrics["mse_std"],
        # "test_mse_sem": model_metrics["mse_sem"],
        # "test_PixCorr_mu": model_metrics["pixel_corr_mu"],
        # "test_PixCorr_std": model_metrics["pixel_corr_std"],
        # "test_PixCorr_sem": model_metrics["pixel_corr_sem"],
        # "test_ssim_mu": model_metrics["ssim_mu"],
        # "test_ssim_std": model_metrics["ssim_std"],
        # "test_ssim_sem": model_metrics["ssim_sem"],
        # "test_lpips_mu": model_metrics["lpips_mu"],
        # "test_lpips_std": model_metrics["lpips_std"],
        # "test_lpips_sem": model_metrics["lpips_sem"],
    }
    
    # Read existing CSV data
    existing_data = []
    with open(csv_file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        for row in reader:
            if row["model_name"] != config['model_type'] or row["subject_id"]!=config['subject_id'][0]:  # Keep all rows except the one with matching model name
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