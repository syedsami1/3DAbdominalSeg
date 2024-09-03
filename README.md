# 3D Segmentation Model for Abdominal Organs

## Overview

This repository contains code and documentation for a 3D segmentation model designed to identify and segment key abdominal organs from CT scans. The primary goal is to segment the Liver, Right Kidney, Left Kidney, and Spleen.

## Dataset

The dataset for this task can be downloaded from the following link:
[CT Abdomen Organ Segmentation Dataset](#)

The dataset includes labeled CT scans. Focus on the following organs:
- Liver
- Right Kidney
- Left Kidney
- Spleen

Refer to the dataset documentation for class IDs corresponding to these organs.

## Model Development

### 1. Model Architecture

- **Architecture Used:** VNet
- **Description:** VNet is a 3D convolutional network designed for volumetric medical imaging tasks. It features an encoder-decoder structure with residual connections, making it suitable for precise segmentation of complex structures in 3D data.

### 2. Training

- **Procedure:** The model was trained using the provided CT scans.
- **Data Splitting:** The dataset was split into training, validation, and testing sets.
- **Training Details:** Includes data augmentation and normalization to enhance model performance.

### 3. Validation and Inference

- **Validation Metrics:** Dice Score, computed for each organ to assess segmentation accuracy.
- **Inference:** The trained model was applied to unseen CT scans to generate segmentations.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/3D-Segmentation-CT-Abdomen.git
   cd 3D-Segmentation-CT-Abdomen
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code:**
   - To train the model:
     ```bash
     python train.py --config config.yaml
     ```
   - To evaluate the model:
     ```bash
     python evaluate.py --model_path path/to/model --data_path path/to/test_data
     ```

4. **Generate 3D Visualizations:**
   - To create a 3D visualization of the predicted segments:
     ```bash
     python visualize.py --model_path path/to/model --data_path path/to/test_data --output_path path/to/output
     ```

## Model Architecture

- **VNet:** A volumetric segmentation network with an encoder-decoder structure and residual blocks, designed to effectively segment 3D medical images.

### Key Architectural Details:
- **Encoder-Decoder Structure:** Captures multi-scale features for accurate segmentation.
- **Residual Blocks:** Enhances feature learning and reduces training time.
- **3D Convolutions:** Efficiently processes volumetric data.

## Training Process

- **Data Preprocessing:** Includes normalization and augmentation techniques.
- **Training Configuration:**
  - Learning Rate: 1e-4
  - Batch Size: 4
  - Epochs: 50

## Validation and Inference

- **Validation Metric:** Dice Score
- **Performance:** Dice Scores for each organ are provided in the evaluation results.

## 3D Visualization

A video demonstrating the 3D rendered segments of the predicted organs can be viewed [here](#). This visualization showcases the segmented Liver, Right Kidney, Left Kidney, and Spleen.

## Final Steps

- Ensure the GitHub repository link is accessible.
- The README file includes all necessary sections.
- The 3D visualization video is included in the README.

