import os
import numpy as np
import librosa
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# emotion mapping
emotion_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

# Librosa is a library used for feature extraction allowing for audio analysis, as without it,
# audio files would be too large to process
# Find features that can be extracted from the dataset: https://librosa.org/doc/latest/feature.html

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None) # audio, sample rate returned
     
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # MFCC/Mel Frequency Cepstral Coefficients, short term power spectrum
    chroma = librosa.feature.chroma_stft(y=y, sr=sr) # pitch
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr) # power spectrogram, mel scale
    # delta1 = librosa.feature.delta(mfcc) # first order difference of MFCC
    # rms = librosa.feature.rms(y=y) # root mean square energy
    
    # Concatenate along feature axis (no averaging to preserve time)
    features = np.concatenate((mfcc, chroma, spectrogram), axis=0)  # (180, T)
    return features


def visualize_audio_features(file_path, output_dir="feature_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    y, sr = librosa.load(file_path, sr=None)
    filename = os.path.basename(file_path).replace('.wav', '')
    
    # Extract features (without averaging for better visualization)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Save individual plots
    fig_mfcc, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'MFCC - {filename}', fontsize=12, fontweight='bold')
    ax.set_ylabel('MFCC Coefficient')
    ax.set_xlabel('Time Frame')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    mfcc_path = os.path.join(output_dir, f"{filename}_mfcc.png")
    plt.savefig(mfcc_path, dpi=150, bbox_inches='tight')
    
    fig_chroma, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(chroma, aspect='auto', origin='lower', cmap='magma')
    ax.set_title(f'Chroma - {filename}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Chroma Bin')
    ax.set_xlabel('Time Frame')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    chroma_path = os.path.join(output_dir, f"{filename}_chroma.png")
    plt.savefig(chroma_path, dpi=150, bbox_inches='tight')
    
    fig_spec, ax = plt.subplots(figsize=(12, 5))
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    im = ax.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='plasma')
    ax.set_title(f'Mel Spectrogram - {filename}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mel Frequency Bin')
    ax.set_xlabel('Time Frame')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    spec_path = os.path.join(output_dir, f"{filename}_spectrogram.png")
    plt.savefig(spec_path, dpi=150, bbox_inches='tight')
    
    print(f"Saved feature plots to {spec_path}")
    plt.close('all')


def load_dataset(data_path):
    X = []
    y = []
    max_T = 0
    
    # First pass: collect features and find max time steps
    for file in os.listdir(data_path):
        if file.endswith(".wav"):
            file_path = os.path.join(data_path, file)
            emotion = file.split("_")[2]
            label = emotion_map[emotion]
            features = extract_features(file_path)  # (180, T)
            X.append(features)
            y.append(label)
            max_T = max(max_T, features.shape[1])
    
    # Second pass: pad to max_T
    X_padded = []
    for features in X:
        T = features.shape[1]
        if T < max_T:
            pad_width = ((0, 0), (0, max_T - T))  # Pad time axis only
            features = np.pad(features, pad_width, mode='constant', constant_values=0)
        X_padded.append(features)
    
    return np.array(X_padded), np.array(y)  # (N, 180, max_T)


if __name__ == "__main__":
    save_to_files = True
    dataset_path = "AudioWAV"
    display_visualizations = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.get_device_name(0))
    print("Load data on device:", device)

    X, y = load_dataset(dataset_path)

    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    
    # Visualize first audio sample if flag is true
    if display_visualizations:
        print("\nGenerating visualizations for first audio sample...")
        audio_files = [f for f in os.listdir(dataset_path) if f.endswith(".wav")]
        if audio_files:
            first_audio_path = os.path.join(dataset_path, audio_files[0])
            visualize_audio_features(first_audio_path)
        else:
            print("No audio files found in the dataset path.")
    
    if save_to_files:
        # Save to .npy files
        np.save("features.npy", X)
        np.save("labels.npy", y)