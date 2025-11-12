# On the Role of Low-Level Visual Features in EEG-Based Image Reconstruction

Official implementation of "On the Role of Low-Level Visual Features in EEG-Based Image Reconstruction" <!-- -accepted at [Conference/Journal Name]-->.

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

### Install required packages
```bash
pip install wandb
pip install einops
pip install open_clip_torch
pip install transformers==4.28.0.dev0
pip install diffusers==0.24.0
pip install braindecode==0.8.1
```

## 2. Downloading Data
In this study, we directly used the preprocessed EEG data and the VAE latents provided by [Li et al.](https://arxiv.org/abs/2403.07721#:~:text=In%20this%20study%2C%20we%20present%20an%20end-to-end[...] which can be downloaded on their [Huggingface](https://huggingface.co/datasets/LidongYang/EEG_Image_decode).
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

### Evaluation on Test Datasets
These scripts will create csv files that store the configuration and the metric values across models and subjects. And the first 30 reconstructions will also be saved.
```bash
# Low-level reconstruction
bash EEG_lowlevel_metrics.sh

# High-level reconstruction
bash EEG_highlevel_metrics.sh

# Two-level reconstruction
bash EEG_final_metrics.sh
```

## Citation

The paper for this code has been published. Please cite the published version (DOI: 10.1109/MLSP62443.2025.11204210) when using this code or the reported results.

Suggested citation (plain text):
On the Role of Low-Level Visual Features in EEG-Based Image Reconstruction. J. Yang et al., 2025. IEEE MLSP. DOI: 10.1109/MLSP62443.2025.11204210

BibTeX template for this work (published):
```bibtex
@inproceedings{yang2025_lowlevel_eeg,
  title = {On the Role of Low-Level Visual Features in EEG-Based Image Reconstruction},
  author = {Yang, J. and Coauthor, A. and Coauthor, B.},
  booktitle = {Proceedings of the 2025 IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year = {2025},
  doi = {10.1109/MLSP62443.2025.11204210},
  url = {https://doi.org/10.1109/MLSP62443.2025.11204210},
  note = {Code: https://github.com/jyang635/EEG_decoding}
}
```

If you used the preprocessed EEG data and VAE latents provided by the external dataset, please cite the original dataset and paper as well. Example entries:

Hugging Face dataset (EEG_Image_decode)
```bibtex
@misc{lidongyang_eeg_image_decode_2024,
  title = {EEG_Image_decode},
  author = {Lidong Yang},
  year = {2024},
  howpublished = {Hugging Face Dataset},
  url = {https://huggingface.co/datasets/LidongYang/EEG_Image_decode}
}
```

ArXiv reference for the dataset / preprocessing (as linked in this README)
```bibtex
@misc{dataset_arxiv_2403_07721,
  title = {Title of the arXiv paper (replace with actual title)},
  author = {Authors (replace with actual author list)},
  year = {2024},
  note = {arXiv:2403.07721},
  url = {https://arxiv.org/abs/2403.07721}
}
```

If you'd like, provide the final publication details (full author list, venue name if different, year) and I can update the BibTeX entries accordingly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [s2560450@ed.ac.uk]