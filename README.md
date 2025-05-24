# On the Role of Low-Level Visual Features in EEG-Based Image Reconstruction

Official implementation of "[On the Role of Low-Level Visual Features in EEG-Based Image Reconstruction]" <!-- -accepted at [Conference/Journal Name]-->.

<!--## Abstract

[Brief description of your paper and its contributions]

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended -->

## 1. Preparing the Virtual Environment

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate BCI
```

<!--### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate the environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate -->

# Install required packages
```bash
pip install wandb
pip install einops
pip install open_clip_torch
pip install transformers==4.28.0.dev0
pip install diffusers==0.24.0
pip install braindecode==0.8.1
```

## 2. Downloading Data
In this study, we directly used the preprocessed EEG data and the VAE latents provided by [Li et al.](https://arxiv.org/abs/2403.07721#:~:text=In%20this%20study%2C%20we%20present%20an%20end-to-end%20EEG-based,embedding%2C%20and%20a%20two-stage%20multi-pipe%20EEG-to-image%20generation%20strategy.), 
which can be downloaded on their [Huggingface](https://huggingface.co/datasets/LidongYang/EEG_Image_decode).
The raw visual stimuli can be downloaded on [OSF](https://osf.io/3jk45/).

### Data Structure

After downloading, your data directories should look like:

```
EEG_data/
├── sub-01/
│   ├── preprocessed_eeg_training.npy
│   ├── preprocessed_eeg_test.npy
├── sub-02/
│   ├── preprocessed_eeg_training.npy
│   ├── preprocessed_eeg_test.npy
```

## 3. Training

### Quick Start
#### High-level pipeline
```bash
# First modify the Config file to speficy data folders
vi data_config.json

# Train the stage-1 high-level models for the 10 subjects
bash EEG_stage1_highlevel.sh --gpu 0 --data_path [your EEG path]

# Train the stage-2 diffusion models for the 10 subjects
bash EEG_stage2_highlevel.sh --gpu 0 --data_path [your EEG path] --save_model
```
#### Low-level pipeline
```bash
bash EEG_stage1_lowlevel.sh --gpu 0 --data_path [your EEG path] --save_model
```

## 4. Metric Computation

### Evaluation on Test Set

```bash
# Low-level reconstruction
bash EEG_lowlevel_metrics.sh

# High-level reconstruction
bash EEG_highlevel_metrics.sh

# Two-level reconstruction
bash EEG_final_metrics.sh
```

### Results Format

Results are saved in JSON format:

```json
{
  "test_accuracy": 0.945,
  "test_f1_score": 0.932,
  "test_precision": 0.941,
  "test_recall": 0.923,
  "per_class_metrics": {
    "class_1": {"precision": 0.95, "recall": 0.92, "f1": 0.93},
    "class_2": {"precision": 0.93, "recall": 0.94, "f1": 0.93}
  }
}
```

## Reproducing Paper Results

To reproduce the main results from the paper:

```bash
# Run full experimental pipeline
bash scripts/reproduce_results.sh

# Or run individual experiments
python train.py --config configs/paper_experiment_1.yaml
python train.py --config configs/paper_experiment_2.yaml
```

## Pre-trained Models

Download pre-trained models:

```bash
# Download all pre-trained models
python scripts/download_models.py

# Download specific model
python scripts/download_models.py --model model_name
```

| Model | Dataset | Accuracy | Download |
|-------|---------|----------|----------|
| Model A | Dataset 1 | 94.5% | [Link](link_to_model) |
| Model B | Dataset 2 | 92.3% | [Link](link_to_model) |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Your Paper Title},
  author={Author1, Author2, Author3},
  journal={Journal/Conference Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your.email@domain.com]

## Acknowledgments

- [Any acknowledgments or credits]
- [Links to related work or dependencies]
