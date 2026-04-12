# ece1513-project

Developed by Group 4:
- Siddharth Joshi
- Jingyu Han
- Tyler Sun

The following explains setup and usage for Group 4's project, Speech Emotion Recognition from Audio using Deep Neural Networks, which aims to classify audio files of human speech with a corresponding human emotion.

## Setup

The scripts for the final model, as well as most of the earlier model iterations, require Python and an environment with the dependencies located in [**requirements.txt**](requirements.txt).

To install all necessary packages in your environment, run in your directory:
```
pip install -r requirements.txt
```

Some earlier models which focused more on preprocessing were developed and are run on .ipynb notebooks. To run these, a platform that supports Jupyter notebooks is required.

## Usage Instructions

(explain how to run final model)

To run the model on any audio sample, including a self-recorded voice clip, use the run_classification.py script:
```
python run_classification.py
```
The script requires the model to be trained at least once, as it takes in a weights file and also reads normalization statistics (mean, standard deviation) from a file produced while running the model on the training dataset. The paths are configured as outputs\best.pt and outputs\normalization_stats.npz, but relative paths can be adjusted as needed. Running this file on a valid .wav file path will produce a predicted emotion class based off the highest class probability, along with its confidence rating.

Earlier versions of the model can be found in the [**Logistic Regression**](Logistic Regression), [**CNN + Attention**](CNN+Attention) and [**CNN + Attention + Delta**](CNN+Attention+Delta) folders. Each of these contain a similar set of files to the final model, including a preprocessing script, model code and training script. To run any of the earlier models, navigate to the respective model directory and run train.py - it is recommended to change your current directory before running to avoid overwritting output files from other models.
