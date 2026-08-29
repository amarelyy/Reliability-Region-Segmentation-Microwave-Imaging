# **Project Documentation: Reliability-Region Segmentation**

## 1. Environment Setup
### Create a virtual environment using Python 3.11
py -3.11 -m venv venv_ai

### Activate the environment
.\venv_ai\Scripts\activate

### Install dependencies (PyTorch with CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy matplotlib joblib scikit-learn

## 2. Data Preparation
Ensure the data/ folder contains the following files from the UM-BMID dataset:
- fd_data_gen_two_s21.pickle
- metadata_gen_two.pickle
- phantom_database.csv

## 3. Pipeline Execution
### Step A: Radar Reconstruction (Physics Baseline)
Run the ablation study to generate radar images using the Snellius Bistatic model. Feel free to adjust number of workers or number of scans. 

To smoke test, use:
`python ablation_runner.py --n-scans 30 --n-jobs 1`
To run all, use:
`python ablation_runner.py --n-jobs 1`

### Step B: U-Net Training
Train the 2-channel U-Net model for reliability-region segmentation.
`python -m unet.training.train`

### Step C: Evaluation & Visualization
Evaluate the trained model and generate reliability maps.

Run evaluation metrics (Dice Score, CoM Error)
`python -m unet.inference.evaluate`

Generate visual comparisons
`python -m unet.inference.visualize`