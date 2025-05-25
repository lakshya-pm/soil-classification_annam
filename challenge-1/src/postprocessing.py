"""

Author: Lakshya Marwaha
Team Name: Individual
Team Members: -
Leaderboard Rank: 48

"""

# Here you add all the post-processing related details for the task completed from Kaggle.

import numpy as np
import torch
import pandas as pd

def multiclass_postprocess(logits, label_map={0: 'Alluvial soil', 1: 'Black soil', 2: 'Red soil', 3: 'Clay soil'}):
    """Convert logits to class predictions and format for submission."""
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    labels = [label_map[p] for p in preds]
    return labels

def save_submission(image_names, labels, output_path):
    df = pd.DataFrame({'image': image_names, 'label': labels})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

def postprocessing():
    print("This is the file for postprocessing")
    return 0
