# ArchieSeamons-CW2-UP2110919-Image_Classification
AI Usage:
- Aspects of Section 1.3 were bug-fixed and generated.
- Tables in the markdowns were pasted from Excel and converted into the correct format.
- Saving both the model and its history for evaluation later on.
- Converting the code in 8.3 to show 5 incorrect images rather than 1.
- Grammarly used to identify typos and grammatical errors in the README file

# Garbage Image Classification - M32895 Bif Data Applicatons CW2
## Student Name: Archie Seamons, Student ID: UP2110919, Year: 2025/26, Examiner: Dr Sergey Yakovlev

## Introduction
This project develops a full Machine Learning (ML) pipeline to classify images of waste into 6 categories using a Convolutional Neural Network (CNN). The pipeline covers every stage from raw-data collection and cleaning to individual real-world predictions. The Garbage Classification dataset from Kaggle is used, which contains 2500* images spanning across 6 classes of household and recyclable waste (cardboard, glass, metal, plastic, and trash).

Two CNN models were produced: a simple baseline and a deeper, optimised version. Both were evaluated and compared, with the best-performing model selected for final testing on unseen data. All steps are documented with explanations, and the reasoning behind each decision is provided throughout.

## Objectives
**Purpose**: To build a reliable image classification system that can identify types of waste in a supported photograph. Such a system could power a smartphone app that helps members of the public sort their recycling, or automate bin-sorting waste management facility.

**Project Goal**: Achieve the highest possible classification accuracy across 6 waste categories, while minimising overfitting so that the model generalises well to new, unseen images.

# ML Pipeline
## 1. Data Collection
The dataset was downloaded from Kaggle and imported into the codespace as a .zip file. The file was then extracted through the use of the terminal. The folder contains 6 sub-folders, one per waste class. Images are in .jpg and .png formats.

A custom `validate_dataset()` function was written to check every image before use. The function:
- Checks that files have a valid extension
- Attempts to open each image using PIL's `.verify()` method to catch any corrupt files
- Reports a per-class summary of the valid and invalid images

Validation results confirmed 2527 valid images and 0 invalid images across all the classes.

The full dataset was then split into train, validation and test subsets:
- 80% Train
- 10% Validation
- 10% Test

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=0)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0)
```

Images were loaded manually from folders using PIL and converted into NumPy Arrays; this is the format the CNN expects. All images were then resized to 100x100 pixels and pixel values rescaled/normalised from `[0, 255]` to `[0.0, 1.0}` by dividing by 255. This step helps the model train faster and with more stability, since large values can cause instabilities. Integer labels were then converted to "one-hot" encoded format using `to_categorical()`.

## 2.EDA
The EDA section explored:
- The distribution of different classes in the three sets (train, validation, and test).
- Image samples show one of each class from the database to display potential similarities and classes that may be confused.
- The mean pixel values across all images within a class, showcasing the typical shape and colour associated with each class.

## 3.Model Building
A CNN was selected for this task because it is specifically designed for image data. Convolutional layers learn spatial features (edges, textures, and shapes) hierarchically across the image, making them far more suitable than a traditional ML model, such as decision trees or logistic regression.

Two models were built and compared:

**Baseline CNN**

| Layer | Configuration | Purpose |
|-------|---------------|---------|
| Conv2D | 32 filters, 3×3, ReLU | Learns low-level features (edges, corners) |
| MaxPool2D | 2×2 | Reduces spatial size, retains dominant features |
| Conv2D | 64 filters, 3×3, ReLU | Learns intermediate features (textures) |
| MaxPool2D | 2×2 | Further spatial reduction |
| Flatten | — | Converts 2D maps to a 1D vector |
| Dense | 128 units, ReLU | Combines learned features |
| Dropout | 0.5 | Reduces overfitting by randomly disabling 50% of neurons |
| Dense | 6 units, Softmax | Outputs a probability for each of the 6 classes |

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

```python
history_base_fit = baseline_model.fit(
    x=X_train,
    y=y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stop]
)
```

**Optimised CNN**

| Change from Baseline | Reason |
|----------------------|--------|
| Added 3rd Conv2D block (128 filters) | Deeper networks can learn more complex, class-discriminating features |
| Added BatchNormalization after each Conv block | Normalises layer activations — stabilises gradients and speeds up convergence |
| Reduced Dropout from 0.5 to 0.4 | Larger network benefits from slightly less aggressive regularisation |
| Increased Dense layer from 128 to 256 units | Greater capacity to combine the richer feature set from 3 conv blocks |

```python
history_opt_fit = optimised_model.fit(
    x=X_train,
    y=y_train,
    epochs=25,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stop]
)
```

Both the models were saved after training using `model.save()`, and training histories were saved as `.json` files so that models could be reloaded without retraining.

## 4.Model Evaluation

Both models were evaluated using the following metrics:

**Accuracy**: The proportion of all images that were correctly classified. The higher the better.

**Precision**: How many of class X predictions actually were class X? Higher precision means fewer false positives.

**Recall**: Of all class X images, what fraction did the model correctly identify? Higher recall means fewer items are missed.

**F1-Score**: The mean of precision and recall. Higher is better. This balances both metrics and is useful for evaluating individual class performance.

**Training Curves**: Accuracy and loss over epochs were plotted for both models. A large gap between training and validation curves can signal overfitting. The optimised model shows a larger gap, indicating it is overfitting more than the baseline.

**Confusion Matrix**: A heatmap of true vs predicted labels. The diagonal (top left to bottom right) shows correct predictions. This matrix reveals that glass and plastic are the most commonly confused classes.

**Baseline CNN Results (Validation Set)***

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| cardboard | 0.97 | 0.93 | 0.95 |
| glass | 0.86 | 0.90 | 0.88 |
| metal | 0.89 | 0.94 | 0.91 |
| paper | 0.88 | 0.96 | 0.92 |
| plastic | 0.94 | 0.79 | 0.86 |
| trash | 0.83 | 0.81 | 0.82 |
| **Overall Accuracy** | | | **0.90** |

*Results may differ after re-teaching the model.

## 5.Model Comparison & Selection

The baseline CNN outperformed the optimised CNN on the validation set. This is a common result when working with small datasets. The optimised model has significantly more parameters, which require more training data to learn effectively. With approximately 2000 training images across 6 classes, the extra complexity works against the model; it begins memorising training data rather than learning general patterns.

The Baseline CNN was therefore selected as the best model for evaluating the final test set.

## 6.Prediction on Unseen Data

The selected baseline CNN was evaluated on the fully held-out test set; these images were never used during training or hyperparameter selection. This gives a true, unbiased estimate of real-world performance.

Individual image predictions were demonstrated using an `index` variable to select specific images from the test set; both a correct and an incorrect prediction were shown, as well as the first 5 incorrect predictions. This showed where the model struggles the most.

# Jupyter Notebook Structure

| Section | Content |
|---------|---------|
| Step 1 | Library imports, dataset overview, data validation, image loading, sample check, array validation, train/val/test split, rescaling and one-hot encoding |
| Step 2 | Class distribution chart, per-set distribution, sample images grid, average image per class |
| Step 3 | Baseline CNN architecture, training with early stopping, model and history saving |
| Step 4 | Training curves (loss and accuracy), confusion matrix and classification report for train/val/test |
| Step 5 | Optimised CNN architecture, training with early stopping, model and history saving |
| Step 6 | Training curves, confusion matrix and classification report for train/val/test |
| Step 7 | Model comparison summary table and bar chart, best model selection |
| Step 8 | Test set evaluation, single correct prediction, single incorrect prediction, grid of 5 incorrect predictions |

# Future Work

In the future, the following improvements should be explored:

**Larger Dataset**: Supplementing the current dataset with additional images, particularly for already small classes, would allow both models to learn more generalisable patterns and likely close the gap between the baseline and optimised models.

**Class Weighting**: Applying class weights during training to compensate for imbalance between classes could improve recall on underrepresented categories.

# Key Libraries and Modules:

- TensorFlow: Deep learning for building the CNN
- NumPy - Allows for efficient array operations
- Pandas - Data handling for classification reports
- Matplotlib - Python's core plotting library
- Seaborn - A statistical visualisation library built on Matplotlib
- Scikit-learn - A general-purpose ML library used for reports and confusion matrices
- PIL - A Python image processing library.
- JSON - Used to save and load training histories in the .Json format.

# Unfixed Bugs

At the time of submission, there are no known bugs. The notebook runs end-to-end without errors so long as the dataset is placed in the correct place.

# Conclusions

This project successfully implemented a complete ML pipeline for waste image classification across 6 categories. A baseline CNN was first trained to establish a performance floor, achieving 90% overall accuracy on the training set. An optimised CNN was developed with a third convolution block, BatchNormalisation and a larger dense layer. However, the baseline CNN outperformed the optimised model, mainly due to the small dataset.

The baseline CNN was then selected as the best model and evaluated on a fully held-out test set to give an honest, real-world performance estimate. Individual predictions were shown, and a grid of misclassified images highlights where the model struggles the most. Future work would prioritise dataset expansion to significantly improve performance.
