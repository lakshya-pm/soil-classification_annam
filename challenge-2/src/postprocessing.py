"""

Author: Lakshya Marwaha
Team Name: Individual
Team Members: -
Leaderboard Rank: 44

"""

import numpy as np
import pandas as pd

def binary_postprocess(decision_scores, threshold=0.0):
    """Convert SVM decision scores to binary predictions."""
    preds = (np.array(decision_scores) > threshold).astype(int)
    return preds

def save_submission(image_names, preds, output_path):
    df = pd.DataFrame({'image': image_names, 'label': preds})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

def postprocessing():
    print("This is the file for postprocessing")
    return 0 