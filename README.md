# Soil Classification Competition Submission

## Overview
This submission contains solutions for two soil classification tasks:

### Task 1: Multi-class Soil Classification
- **Architecture**: ConvNeXt-Base ensemble with 5-fold cross-validation
- **Performance**: 1.0000 F1-score
- **Classes**: Alluvial, Red, Black, Clay soil types
- **Innovation**: Weighted loss function for class imbalance

### Task 2: Binary Soil Detection  
- **Method**: One-Class SVM with ConvNeXt feature extraction
- **Performance**: 0.8965 F1-score
- **Challenge**: Binary classification with single-class training data
- **Innovation**: Ensemble of multiple nu parameters

## Results Summary
- Challenge 1: Rank 48 (1.0000 F1-score)
- Challenge 2: Rank 44 (0.8965 F1-score)


## Setup and Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/lakshya-pm/soil-classification-challenge.git
cd soil-classification-challenge
```

### 2. Install Dependencies
It is recommended to use a virtual environment (e.g., `venv` or `conda`):

```bash
# Using pip and venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements for both challenges
pip install -r challenge-1/requirements.txt
pip install -r challenge-2/requirements.txt
```

Or, if you use conda:
```bash
conda create -n soil-classification python=3.8
conda activate soil-classification
pip install -r challenge-1/requirements.txt
pip install -r challenge-2/requirements.txt
```

### 3. Prepare the Data
- Place the provided datasets in the following folders:
  - For **Challenge 1**:  
    `challenge-1/data/soil_classification-2025/`  
    (with `train/`, `test/`, `train_labels.csv`, `test_ids.csv`)
  - For **Challenge 2**:  
    `challenge-2/data/soil_competition-2025/`  
    (with `train/`, `test/`, `train_labels.csv`, `test_ids.csv`)

### 4. Run the Notebooks
You can run the Jupyter notebooks for each challenge:

#### Challenge 1
```bash
cd challenge-1/notebooks
jupyter notebook
# Open "Training and Inference.ipynb" and run all cells
```

#### Challenge 2
```bash
cd challenge-2/notebooks
jupyter notebook
# Open "soil-classification-2.ipynb" and run all cells
```

### 5. Run as Python Scripts (Optional)
If you have modularized scripts, you can run them as follows:

```bash
# Example for training (if scripts are available)
python challenge-1/src/train.py
python challenge-2/src/train.py

# Example for inference (if scripts are available)
python challenge-1/src/inference.py
python challenge-2/src/inference.py
```

### 6. Output
- Submission files will be generated in the respective directories as `submission.csv` or `optimized_submission.csv`.
- Check the notebook outputs for performance metrics and leaderboard scores.

### 7. Troubleshooting
- Ensure all dependencies are installed.
- Check that the data paths match the expected structure.
- For GPU acceleration, ensure CUDA is available and PyTorch is installed with GPU support.

---

## License
This project is licensed under the MIT License.

## Contact
For any issues, please refer to the code comments or raise an issue on the GitHub repository.
