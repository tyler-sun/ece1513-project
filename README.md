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
Some earlier models which focused more on preprocessing were developed and are run on .ipynb 
notebooks, which can be found in [**CNN+DataPreprocesses**](CNN+DataPreprocesses). To run 
these, a platform that supports Jupyter notebooks is required, such as Google Colab or 
JupyterLab. The notebooks rely on the following key libraries:

- **librosa** — audio loading, Mel spectrogram, delta/delta-delta feature extraction
- **numpy** — tensor manipulation and numerical operations
- **essentia** — pitch extraction via PredominantPitchMelodia (earlier iterations only)
- **audiomentations** — waveform-level augmentation (time stretch, pitch shift, noise, shift)
- **tensorflow / keras** — CNN, CNN-LSTM, ResNet-LSTM, and fusion model training
- **torch / torchvision** — final model training pipeline
- **scikit-learn** — train/val/test splitting and label encoding
- **matplotlib / seaborn** — learning curves and confusion matrix visualisation
- **pandas** — metadata handling and dataset indexing
- **tqdm** — progress tracking during batch feature extraction

The notebooks also require access to the CREMA-D dataset, which can be downloaded directly via the Kaggle API using the dataset ID `ejlok1/cremad`. Google Drive is used for saving 
processed tensors and model checkpoints across sessions.

The dataset used is the audio files of CREMA-D, which are all in .wav formatting and can be downloaded from Kaggle: https://www.kaggle.com/datasets/ejlok1/cremad. Store these files in a directory which can be read by the training scripts (ex. 'data\crema'). Alternatively, the dataset can be accessed with a Kaggle API key if accessible.

Some earlier models that were tested using data focused more on preprocessing were developed and are run on .ipynb notebooks, which can be found in [**CNN+DataPreprocesses**](CNN+DataPreprocesses). To run these, a platform that supports Jupyter notebooks is required, such as Google Colab.

## Usage Instructions
### Data Preprocessing Notebooks
All preprocessing notebooks are designed to run on Google Colab and require a Kaggle API key (`kaggle.json`) for dataset access. Each notebook mounts Google Drive to save outputs 
across sessions.

**DataPreprocess3D**
Connects directly to Kaggle to download the CREMA-D dataset and processes all raw `.wav` files into three-channel tensors (Mel spectrogram, MFCC, and pitch). The processed tensors 
are saved to Google Drive for use in subsequent notebooks. To run, upload your `kaggle.json` when prompted and ensure your Google Drive has sufficient storage for the full tensor dataset.

**DataPreprocess5D**
Loads the three-channel tensors saved by DataPreprocess3D from Google Drive and extracts additional features (Chroma, Spectral Contrast, and Tonnetz) to produce five-channel tensors. 
The best model weights and confusion matrix from each training run are saved back to Google Drive.

**DataPreprocessFinal**
Downloads the CREMA-D dataset fresh from Kaggle and applies the final feature extraction pipeline directly to the raw audio, producing three-channel Mel, delta, and delta-delta 
tensors. For each training run, model weights, learning curve plots, confusion matrix, and evaluation results are saved to Google Drive. To run, upload your `kaggle.json` when prompted 
and ensure Google Drive is mounted before executing the training cells.

### Final Model
Download the CREMA-D dataset from Kaggle and place it in the following directory: `data/crema`. Ensure that the `src` folder is located at the same level as the `data` folder so that all scripts can correctly access the dataset.
To train the final model, run:
```
python train.py
```
All results will be saved in the `outputs` directory, including: trained model weights, training curves and plots, and evaluation results (accuracy, F1-score, etc.).

To run the model on any audio sample, including a self-recorded voice clip if desired, use the run_classification.py script:
```
python run_classification.py
```
The script requires the model to be trained at least once, as it takes in a weights file and also reads normalization statistics (mean, standard deviation) from a file produced while running the model on the training dataset. The paths are configured as outputs\best.pt and outputs\normalization_stats.npz, but relative paths can be adjusted as needed. Running this file on a valid .wav file path will produce a predicted emotion class based off the highest class probability, along with its confidence rating. As the model is trained off CREMA-D samples, differences in recording quality, background noise, and other acoustic factors may impact the accuracy of the prediction.

### Earlier Models Tested

Earlier versions of the model can be found in the [**Logistic Regression**](<Logistic Regression>), [**CNN + Attention**](CNN+Attention) and [**CNN + Attention + Delta**](CNN+Attention+Delta) folders. Each of these contain a similar set of files to the final model, including a preprocessing script, model code and training script. To run any of the earlier models, navigate to the respective model directory and run train.py - it is recommended to change your current directory before running to avoid overwritting output files from other models. For example, to run the logistic regression baseline:
```
 cd '.\Logisitic Regression\'
 python .\train.py
```
