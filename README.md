# Facial Landmark Detection

A project focused on facial landmark detection and evaluation using pre-trained models.

## Methods

- **dlib** – HOG face detector and 68-point facial landmark predictor
- **MediaPipe** – Face Mesh with 468 facial landmarks

## Dataset

WFLW (Wide Facial Landmarks in-the-wild), containing images annotated with 98 facial landmarks.

## Evaluation

- Normalized Mean Error (NME)
- Region-based NME

## Project Structure

```text
src/
├── landmark_mapping.py
├── evaluation.py
└── test_mapping.py

data/       # Datasets
models/     # Pre-trained models
```

## Project Status

Preliminary implementation and testing completed. Full dataset evaluation and further analysis are planned for the next stage.