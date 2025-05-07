import os
import numpy as np
import torch
from torch import nn,Tensor
from PIL import Image
# from matplotlib import pyplot as plt
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


# def positive_images(images):
#     # Assuming the input is with shape [batch, channel, H, W]
#     flattened_images = images.view(images.shape[0], images.shape[1], -1)
#     min_values, _ = torch.min(flattened_images, dim=2, keepdim=True)
#     max_values, _ = torch.max(flattened_images, dim=2, keepdim=True)
    
#     # Reshape min_values and max_values to (batch_size, channel, 1, 1) for broadcasting
#     min_values = min_values.view(images.shape[0], images.shape[1], 1, 1)
#     max_values = max_values.view(images.shape[0], images.shape[1], 1, 1)
    
#     # Shift the values of the images so that all values are positive
#     shifted_images = (images - min_values) / (max_values - min_values)
#     return shifted_images

class CustomImageDataset(Dataset):
    def __init__(self, images, responses,data_index, nrep,transform=None, image_zero=False):
        self.images = images
        self.responses = responses
        self.transform = transform
        self.image_zero = image_zero
        self.data_index = data_index
        self.nrep = nrep
        if image_zero:
            print('Normalize images to -1 and 1')

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        # img_idx=idx//4
        img_idx=self.data_index[idx]
        image=self.images[img_idx//self.nrep]
        # image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        response = self.responses[idx]
        if self.image_zero:
            image = 2 * image - 1
        return image, response
    
class Latent_ImageDataset(Dataset):
    def __init__(self, image_latents, responses,data_index,nrep): 
        self.image_latents = image_latents
        self.responses = responses
        self.data_index = data_index
        self.nrep = nrep
    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        # img_idx=idx//4

        img_idx=self.data_index[idx]
        latents=self.image_latents[img_idx//self.nrep ]
        # image = self.images[idx]
        response = self.responses[idx]
        return latents, response

def load_subject_data(subject_id, eeg_dir, img_metadata, start_time=None, end_time=None, 
                      desired_channels=['All'], num_images=None,training=True,average=True):
    # Load image metadata
    # Load EEG data
    eeg_parent_dir = os.path.join(eeg_dir, f'{subject_id}')

    if training:
        eeg_data = np.load(os.path.join(eeg_parent_dir, 'preprocessed_eeg_training.npy'), allow_pickle=True)
        eeg_data_tensor = torch.tensor(eeg_data['preprocessed_eeg_data'], dtype=torch.float32)
        # time_indices = np.where((eeg_data['times'] >= start_time) & (eeg_data['times'] <= end_time))[0]
    else:
        eeg_data = np.load(os.path.join(eeg_parent_dir, 'preprocessed_eeg_test.npy'), allow_pickle=True)
        eeg_data_tensor = torch.tensor(eeg_data['preprocessed_eeg_data'], dtype=torch.float32)
        

    eeg_data['times']=eeg_data['times'][50:]
    time_indices = np.where((eeg_data['times'] >= start_time) & (eeg_data['times'] <= end_time))[0]
    # print(f"Time indices: {time_indices}")
    eeg_data_tensor = eeg_data_tensor[:, :, :, time_indices]


    if num_images is None:
        num_images = len(img_metadata['train_img_concepts'])
    if desired_channels[0] == 'All':
        desired_channels = eeg_data['ch_names']
    channel_indices = sorted([eeg_data['ch_names'].index(ch) for ch in desired_channels])
    eeg_data_tensor = eeg_data_tensor[:num_images, :, channel_indices, :]
    if average:
        # Average across the time dimension
        eeg_data_tensor = eeg_data_tensor.mean(dim=1)
        nrep=1
    else:
        # Make use of every trial
        nrep = eeg_data_tensor.shape[1]
        eeg_data_tensor = eeg_data_tensor.view(-1, eeg_data_tensor.shape[2], eeg_data_tensor.shape[3])

    return eeg_data_tensor,nrep

def load_multiple_subjects(subject_ids,eeg_dir, img_dir,img_metadata, desired_channels,start_time=-0.2,
                           end_time=0.8, num_images=None,image_size=None,compressor=None,training=True,average=True):
    # Load EEG data for multiple subjects
    all_eeg_data = []
    # all_test_data = [] 
    eeg_index= []
    # test_index = []
    print(f'{len(subject_ids)} subjects are loaded')

    for subject_id in subject_ids:
        eeg_data,nrep = load_subject_data(subject_id, eeg_dir, img_metadata, start_time, end_time, desired_channels, 
                                     num_images,training=training,average=average)
        all_eeg_data.append(eeg_data)
        eeg_index=eeg_index+list(range(len(eeg_data)))
    # Concatenate the data from all subjects
    all_eeg_data = torch.cat(all_eeg_data, dim=0)

    if num_images is None:
        num_images = len(img_metadata['train_img_concepts'])

    
    if compressor == 'VAE':
        if training:
            image_latent=torch.load(os.path.join(img_dir, 'train_image_latent_512.pt'))['image_latent']
            image_latent=image_latent[:num_images,:,:,:]
        else:
            image_latent=torch.load(os.path.join(img_dir, 'test_image_latent_512.pt'))['image_latent']
        
        dataset = Latent_ImageDataset(image_latent, all_eeg_data,eeg_index,nrep)
    elif compressor == 'CLIP':
        if training:
            image_latent=torch.load(os.path.join(img_dir, 'ViT-H-14_features_train.pt'))['img_features']
            image_latent=image_latent[:num_images,:]
        else:
            image_latent=torch.load(os.path.join(img_dir, 'ViT-H-14_features_test.pt'))['img_features']
        
        dataset = Latent_ImageDataset(image_latent, all_eeg_data,eeg_index,nrep)

    else:
        if image_size is not None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
        if training:
            images_tensors = [Image.open(os.path.join(img_dir, 'training_images', concept, img_file)) 
                                    for concept, img_file in zip(img_metadata['train_img_concepts'][:num_images], img_metadata['train_img_files'])]
        else:
            images_tensors = [Image.open(os.path.join(img_dir, 'test_images', concept, img_file)) 
                                for concept, img_file in zip(img_metadata['test_img_concepts'], img_metadata['test_img_files'])]
        dataset = CustomImageDataset(images_tensors, all_eeg_data,eeg_index, transform=transform,nrep=nrep)

    if training:
        # Create a dataset for the validation
        num_images = all_eeg_data.shape[0]
        num_validation_samples = int(0.2 * num_images)
        indices = list(range(num_images))
        np.random.shuffle(indices)
        validation_indices = indices[:num_validation_samples]
        train_indices = indices[num_validation_samples:]
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        print(f"Training data: {len(train_dataset)} samples")
        print(f"Validation data: {len(validation_dataset)} samples")
        return train_dataset, validation_dataset, dataset.responses.shape[2]
    else:
        print(f"Test data: {len(dataset)} samples")
        return dataset, dataset.responses.shape[2]
    
