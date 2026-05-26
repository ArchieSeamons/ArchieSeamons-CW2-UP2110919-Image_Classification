# ArchieSeamons-CW2-UP2110919-Image_Classification
AI Usage:
- Aspects of Section 1.3 were bug fixed and generated.
- Tables in the markdowns were pasted from excel and converted into the correct format.
- Saving both the model and its history for evaluation later on.
- Converting the code in 8.3 to show 5 incorrect images rather than 1.

# Garbage Image Classification - M32895 Bif Data Applicatons CW2
## Student Name: Archie Seamons, Student ID: UP2110919, Year: 2025/26, Examiner: Dr Sergey Yakovlev

## Introduction
This project develops a full Machine Learning (ML) pipeline to clasify images of waste into 6 categories using a Convolutional Neural Network (CNN). The pipeline covers every stage from raw-data collection and cleaning to individual real-world predictions. The Garbage Classsification dataset from Kaggle is used, which contains 2500* images spanning across 6 classess of household and recyclable waste (cardboard, glass, metal, plastic, and trash).

Two CNN models were produced: a simple baseline and a deeper, optimised version. Both were evaluated and compared, with the best performing model selected for final testing on unseen data. All steps are documented with explanations and the reasoning behind each decision is provided throughout.

## Objectives
**Purpose**: To build a reliable imageclassification system that can identify tyopes of waste in a supported photograph. Such system could power a smartphone app that helps members of the public sort their recycling, or automate bin-sorting waste management facility.

**Project Goal**: Achieve the highest possible classification accuracy across 6 waste categories, while minimising overfitting so that the model generalises well to new, unseen images.

# ML Pipeline
## 1. Data Collection
The dataset was downloaded from kaggle and improted into the codespace as a .zip file. The file was then extracted through the use of the terminal. The folder contains 6 sub-folders, one per waste class. Images are in .jpg and .png formats.

A custom `validate_dataset()` function was writtent o check every image before use. The function:
- Checks that files have a valid extension
- Attempts to open each image using PIL's `.verify()` method to catch any corrupt files
- Reports a per-class summary of the valid and invalid images

Validation results confirmed 2527 valid images and 0 invalid images across all the classes.

The full dataset was the split into train, validation and test subsets:
- 80% Train
- 10% Validation
- 10% Test

'''
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=0)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
'''