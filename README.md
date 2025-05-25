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
