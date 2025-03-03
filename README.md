# Pool Detection in Satellite Images

## Table of Contents

- [Pool Detection in Satellite Images](#pool-detection-in-satellite-images)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Project Structure](#project-structure)
  - [Installation Requirements](#installation-requirements)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Usage Guide](#usage-guide)
    - [1. OpenCV-Based Approach](#1-opencv-based-approach)
      - [Running the OpenCV Approach:](#running-the-opencv-approach)
    - [2. U-Net Deep Learning Approach](#2-u-net-deep-learning-approach)
      - [Training the Model:](#training-the-model)
      - [Testing the Model:](#testing-the-model)
    - [3. Colab Notebook for Testing](#3-colab-notebook-for-testing)
  - [Results](#results)
    - [1. OpenCV-Based Approach](#1-opencv-based-approach-1)
      - [Example Results:](#example-results)
      - [Limitations of the OpenCV Approach:](#limitations-of-the-opencv-approach)
    - [2. U-Net Deep Learning Approach](#2-u-net-deep-learning-approach-1)
      - [Example Results:](#example-results-1)
      - [Limitations of the U-Net Approach:](#limitations-of-the-u-net-approach)


## Description

This project explores two approaches for detecting swimming pools in satellite images:  

1. **OpenCV-Based Approach**: Uses color segmentation and contour detection to extract pool regions based on their distinctive blue color.  
   
2. **U-Net Deep Learning Approach**: Trains a neural network for semantic segmentation to generate precise pool masks.  

The goal is to develop an efficient method for identifying pools in aerial imagery. The repository includes dataset preprocessing, training scripts, and inference notebooks for both methods. 

## Project Structure

```bash
C:.
├───Colab_Roboflow_Model
│   └───output
├───OpenCV_Approach
│   ├───output
│   └───swimming_pool
└───UNet_Approach
    ├───dataset
    │   ├───images
    │   ├───labels
    │   └───masks
    ├───notebooks
    └───output
```

## Installation Requirements

### Prerequisites

Ensure you have the following installed before proceeding:

- Python 3.8+
- pip
- Git (for cloning the repository)

### Installation Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/K-Oussama/Swimming-Pool-Detection.git
   cd Swimming-Pool-Detection

2. **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  
    # On Windows use: venv\Scripts\activate

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt

## Usage Guide

### 1. OpenCV-Based Approach

This method uses OpenCV for color-based segmentation and contour detection.

#### Running the OpenCV Approach:
- Navigate to the `OpenCV_Approach` folder.
- Open and run the provided Jupyter Notebook to process images.
- The processed output (segmented pools) will be saved in the `output/` folder.

### 2. U-Net Deep Learning Approach

This method leverages a U-Net model for semantic segmentation of swimming pools.

#### Training the Model:
- Navigate to `UNet_Approach/notebooks/`.
- Open the training notebook and run all cells.
- The model will be trained on the dataset in `UNet_Approach/dataset/`.

#### Testing the Model:
- Once trained, the model can be used for inference.
- Load the trained model in the provided inference notebook and test it on new images.
- Processed images, masks, and model predictions will be saved in the respective `output/` folders.

### 3. Colab Notebook for Testing

A Google Colab notebook is included for easy model testing:
- Located in `Colab_Testing/`
- Upload a trained model and test it on new images.


## Results

### 1. OpenCV-Based Approach

The OpenCV approach uses color-based segmentation and contour detection to identify swimming pools in images. The method extracts blue regions, often representing pools, and finds contours to precisely highlight the pool areas.

#### Example Results:
Here are some examples of the results from the OpenCV-based approach:


<table>
  <tr>
    <td><strong>Input Image</strong></td>
    <td><strong>Detected Pool Contours</strong></td>
  </tr>
  <tr>
    <td><img src="OpenCV_Approach/swimming_pool/000000079.jpg" width="400"></td>
    <td><img src="OpenCV_Approach/output/output_image.jpg" width="400"></td>
  </tr>
</table>

#### Limitations of the OpenCV Approach:
- **Color Sensitivity:** The method relies on detecting the blue color of pools. This means it may not work well with pools that have different colors or pools that blend with the surroundings.
- **Background Interference:** Background noise, such as reflections or other blue objects in the image, can interfere with the segmentation.
- **Simplistic Contours:** While this approach works for simple shapes, it struggles with more irregular pool shapes.

---

### 2. U-Net Deep Learning Approach

The U-Net approach uses a deep learning model for semantic segmentation, providing more robust and flexible segmentation of swimming pools. This method is able to detect pools regardless of their color and is more resilient to different backgrounds.

#### Example Results:
Here are some examples of the results from the U-Net approach:

<table>
  <tr>
    <td><strong>Input Image</strong></td>
    <td><strong>Predicted Mask</strong></td>
    <td><strong>Detected Pool with Contours</strong></td>
  </tr>
  <tr>
    <td><img src="UNet_Approach/output/original_image.jpg" ></td>
    <td><img src="UNet_Approach/output/predicted_mask.jpg" ></td>
    <td><img src="UNet_Approach/output/contour_overlay.jpg" ></td>
  </tr>
</table>


#### Limitations of the U-Net Approach:
- **Training Data Dependency:** The model's performance is heavily dependent on the quality and variety of the training dataset. If the dataset is limited or biased, the model may not generalize well to unseen images.
- **Computational Resources:** Training the U-Net model can be resource-intensive, requiring a good GPU for faster processing and training.
- **Overfitting:** If the model is not properly tuned or if the dataset is not diverse enough, there is a risk of overfitting, which can reduce the model’s ability to generalize.

---

Both approaches have their strengths and limitations. The OpenCV method is simpler and faster, but it lacks the flexibility and robustness of the U-Net deep learning model, which provides more accurate results but requires a more complex setup.


🏊‍♂️

