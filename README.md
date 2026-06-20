# waste-classification-project

A binary image classifier that sorts waste images into **Organic** or **Recyclable** categories, built using transfer learning with a pretrained MobileNetV2 model.

## Overview

Manual waste sorting is slow and error-prone. This project uses a convolutional neural network to automatically classify waste images, which could support smarter recycling systems.

## Dataset

- **Source**: [Waste Classification Data (Kaggle)](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- **Classes**: Organic (O), Recyclable (R)
- **Training images**: 22,500+
- **Test images**: 2,513

## Approach

- Used **MobileNetV2** (pretrained on ImageNet) as a frozen feature extractor
- Added a custom classification head: `GlobalAveragePooling2D → Dense(1, sigmoid)`
- Trained only the new head, keeping the pretrained base frozen
- Loss: Binary Cross-Entropy | Optimizer: Adam

## Results

- **Test Accuracy**: 89.3%
- **Test Loss**: 0.2731

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Matplotlib

## How to Run

1. Open `Image_class.ipynb` in Google Colab
2. Download the dataset from the Kaggle link above and place it in the expected directory structure
3. Run all cells in order (GPU runtime recommended for faster training)

## Future Improvements

- Add data augmentation to improve generalization
- Fine-tune deeper MobileNetV2 layers instead of keeping the entire base frozen
- Expand to multi-class waste categories (plastic, paper, metal, etc.) instead of binary
