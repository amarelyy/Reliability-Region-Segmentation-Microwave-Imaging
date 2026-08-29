# **Project Documentation: Reliability-Region Segmentation**

## 1. Environment Setup
### Create a virtual environment using Python 3.11
`py -3.11 -m venv venv_ai`

### Activate the environment
`.\venv_ai\Scripts\activate`

### Install dependencies 
`pip install -r requirements.txt`

## 2. Data Preparation
Ensure the data/ folder contains the following files from the UM-BMID dataset:
- fd_data_gen_two_s21.pickle
- metadata_gen_two.pickle
- phantom_database.csv

## 3. Pipeline Execution
### Step A: Radar Reconstruction (Physics Baseline)
Run the ablation study to generate radar images using the Snellius Bistatic model. Feel free to adjust number of workers or number of scans. 

To smoke test, use:\
`python ablation_runner.py --n-scans 30 --n-jobs 1`
To run all, use:\
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

## 4. Project Structure
- src/: Core physics pipeline (Snellius delay, signal processing).
- unet/: Deep learning modules (Dataset, Model, Training, Inference).
- data/: Raw UM-BMID dataset and metadata.
- results/: Output CSVs from radar ablation studies.
- unet/checkpoints/: Saved U-Net model weights (.pth).