# Mask-Detector-DL-OpenCV
Deep Learning based Mask Detector using OpenCV for Python. Convolution Neural Network trained model using Keras on Tensorflow. 

## Prerequisites
  - Python (3.6)
  - OpenCV (4.2.0)
  - Keras  (2.3.1)
  - Tensorflow (1.14.0)

## Dataset
The dataset for the project was courtesy of [prajnasb](https://github.com/prajnasb/observations/tree/master/experiements). Consisits of 690 images with mask and 686 images without mask.

## Processing
The dataset images were preprocessed into 100x100 grayscale images and data converted into numpy arrays for training neural
network.

## Model Training
The model was trained with on a 2 Conv Layer CNN architecture with Keras on Tensorflow.

## Face Detection
Region of Interest - face, in this case, was isolated using Haar Cascade Classifiers. A primitive method to 
detect faces from given frame and then passing it through trained model for predictions.

## Execute Code
  - `processData.py` to save numpy array data of images and labels in 'savedData' folder
  - `trainModel.py` to train your neural network and auto-save model in 'model' folder
  - `mainDetect.py` to run real time detection

## Other Training Methods

  - Using VGGNet as a base model for better accuracy in detection.
  - Training models on PyTorch.

## Developments
The next commit will use Caffe as a pretrained model to detect faces. A deep learning approach to isolate and grab ROI from the frame.
