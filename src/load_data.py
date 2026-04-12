import os
import random
import numpy as np
import librosa


# Map CREMA-D emotion codes to integer labels
emotion_map = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5,
}


def parse_cremad_file(filename: str):
    """
    Parse a CREMA-D filename such as:
        1001_DFA_ANG_XX.wav

    Returns:
        speaker_id (int), emotion_label (int)
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None

    try:
        speaker_id = int(parts[0])
    except ValueError:
        return None, None

    emotion_code = parts[2]
    if emotion_code not in emotion_map:
        return None, None

    return speaker_id, emotion_map[emotion_code]


def _pad_or_truncate(feat: np.ndarray, target_len: int):
    """
    Pad or truncate feature along the time axis so every sample
    has the same number of frames.
    """
    cur_len = feat.shape[1]

    if cur_len < target_len:
        pad = target_len - cur_len
        feat = np.pad(feat, ((0, 0), (0, pad)), mode="constant")
    elif cur_len > target_len:
        feat = feat[:, :target_len]

    return feat


def load_audio(
    file_path: str,
    sr: int = 16000,
    max_samples: int | None = None,
):
    """
    Load waveform, resample, normalize peak amplitude,
    and optionally pad/truncate raw waveform length.
    """
    y, _ = librosa.load(file_path, sr=sr)

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    if max_samples is not None:
        if len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)), mode="constant")
        else:
            y = y[:max_samples]

    return y.astype(np.float32)


def augment_audio(y, sr=16000):
    """
    Apply light waveform augmentation.

    Strategy:
    - 50% chance of leaving waveform unchanged
    - otherwise apply exactly ONE augmentation
    """
    if random.random() < 0.5:
        return y

    def time_shift(sig):
        shift = int(sr * random.uniform(-0.1, 0.1))
        return np.roll(sig, shift)

    def add_noise(sig):
        noise_amp = random.uniform(0.001, 0.004) * (np.max(np.abs(sig)) + 1e-8)
        return sig + noise_amp * np.random.randn(len(sig))

    def vol_scale(sig):
        return sig * random.uniform(0.85, 1.15)

    def pitch_shift(sig):
        n_steps = random.uniform(-1.5, 1.5)
        return librosa.effects.pitch_shift(sig, sr=sr, n_steps=n_steps)

    fn = random.choice([time_shift, add_noise, vol_scale, pitch_shift])

    try:
        y_aug = fn(y)
    except Exception:
        y_aug = y

    y_aug = np.asarray(y_aug, dtype=np.float32)
    peak = np.max(np.abs(y_aug)) + 1e-8
    y_aug = y_aug / peak
    return y_aug


def extract_logmel_3ch_from_waveform(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    max_len: int = 360,
):
    """
    Extract 3-channel feature representation:
        channel 0: log-mel spectrogram
        channel 1: delta
        channel 2: delta-delta

    Output shape: (3, n_mels, max_len)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    logmel = _pad_or_truncate(logmel, max_len)
    delta = _pad_or_truncate(delta, max_len)
    delta2 = _pad_or_truncate(delta2, max_len)

    feat = np.stack([logmel, delta, delta2], axis=0).astype(np.float32)
    return feat


def extract_logmel_3ch(
    file_path: str,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    max_len: int = 360,
):
    """
    Convenience wrapper:
    load waveform from file, then extract 3-channel features.
    """
    y = load_audio(file_path, sr=sr)
    return extract_logmel_3ch_from_waveform(
        y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        max_len=max_len,
    )


def build_metadata(data_path: str):
    """
    Collect file paths and labels from CREMA-D directory.

    Returns:
        paths, labels, speakers, files
    """
    paths, labels, speakers, files = [], [], [], []

    wav_files = sorted([f for f in os.listdir(data_path) if f.endswith(".wav")])

    for fname in wav_files:
        speaker_id, label = parse_cremad_file(fname)
        if speaker_id is None:
            continue

        path = os.path.join(data_path, fname)
        paths.append(path)
        labels.append(label)
        speakers.append(speaker_id)
        files.append(fname)

    return (
        np.array(paths),
        np.array(labels, dtype=np.int64),
        np.array(speakers, dtype=np.int64),
        np.array(files),
    )


def build_cache(
    data_path: str,
    cache_dir: str = "cache",
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    max_len: int = 360,
):
    """
    Precompute clean features once and save them to disk.

    This cache is used for:
    - validation / test
    - clean train-time evaluation
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_name = f"cremad_logmel3ch_sr{sr}_mels{n_mels}_len{max_len}.npz"
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        speakers = data["speakers"]
        files = data["files"]
        return X, y, speakers, files

    paths, y, speakers, files = build_metadata(data_path)

    X = []
    for path in paths:
        feat = extract_logmel_3ch(
            path,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            max_len=max_len,
        )
        X.append(feat)

    X = np.array(X, dtype=np.float32)

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        speakers=speakers,
        files=files,
    )
    print(f"Saved cached dataset to {cache_path}")

    return X, y, speakers, files


if __name__ == "__main__":
    DATA_PATH = "data/crema"
    X, y, speakers, files = build_cache(DATA_PATH)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("speakers shape:", speakers.shape)
    print("example file:", files[0] if len(files) > 0 else "none")