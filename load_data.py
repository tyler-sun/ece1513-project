import os
import numpy as np
import librosa

# emotion mapping
emotion_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # make all samples same size
    mfcc = np.mean(mfcc.T, axis=0)
    
    return mfcc


def load_dataset(data_path):
    X = []
    y = []

    for file in os.listdir(data_path):
        if file.endswith(".wav"):
            file_path = os.path.join(data_path, file)

            # extract label
            emotion = file.split("_")[2]
            label = emotion_map[emotion]

            # extract features
            features = extract_features(file_path)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset("data/crema")

    print("X shape:", X.shape)
    print("y shape:", y.shape)