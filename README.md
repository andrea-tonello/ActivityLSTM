# Human Activity Recognition with LSTM

This repository contains an implementation from scratch of a Long Short-Term Memory neural network for Human Activity Recognition (HAR) using the [UCI HAR dataset](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones). The goal is to classify six different physical activities, based on sensor signals from a smartphone.

## Dataset

The dataset was collected from 30 subjects performing the six aforementioned activities while carrying a waist-mounted smartphone with embedded accelerometer and gyroscope sensors. Each record contains 561 features derived from sensor signals.

- **Source:** [Kaggle - Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
- **Activities:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- **Sensors:** 3-axial linear acceleration and angular velocity (50 Hz update)
- **Train Size:** 7352 $\times$ 563 (70%)
- **Validation Size:** 1473 $\times$ 563 (15%)
- **Test Size:** 1473 $\times$ 563 (15%)

## Model

This project implements from scratch an LSTM-based model to capture the temporal patterns in sequential sensor data.

### Model Architecture

- **Input:** Time-series segments of sensor data
- **Layers:**
  - 3 LSTM Layers (128, 64, 32) with dropout 0.3
  - 2 final Fully Connected layers with softmax
- **Output:** One of the six activity classes

## Results

- **Test Accuracy:** 91.4%

