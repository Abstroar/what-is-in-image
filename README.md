## Image Captioning Model

Overview

This project implements an image captioning model that combines visual features from images with text sequences (captions) using a hybrid LSTM-Decoder architecture.

## Model Architecture

The model consists of:

- Image Feature Extractor: Takes pre-extracted 4096-dimensional image features from a CNN.

- Text Feature Extractor (LSTM): Processes tokenized captions and learns sequential dependencies.

- Decoder: Merges the image and text features and predicts the next word in the caption.

## Dataset
- Flickr8k
