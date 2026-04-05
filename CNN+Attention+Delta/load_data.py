import os
import random
import numpy as np
import librosa


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
    CREMA-D example:
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
    cur_len = feat.shape[1]

    if cur_len < target_len:
        pad = target_len - cur_len
        feat = np.pad(feat, ((0, 0), (0, pad)), mode="constant")
    elif cur_len > target_len:
        feat = feat[:, :target_len]

    return feat


def extract_logmel_3ch(
    file_path: str,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    max_len: int = 360,
    use_speed_perturb: bool = False,
):
    """
    Returns 3-channel feature:
        channel 0: log-mel
        channel 1: delta
        channel 2: delta-delta

    Output shape: (3, n_mels, max_len)
    """
    y, _ = librosa.load(file_path, sr=sr)

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    if use_speed_perturb:
        speed = random.choice([0.9, 1.0, 1.1])
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed)

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
    Precompute all base features and save to cache.

    Returns:
        X: (N, 3, 128, T)
        y: (N,)
        speakers: (N,)
        files: (N,)
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

    X, y, speakers, files = [], [], [], []

    wav_files = sorted([f for f in os.listdir(data_path) if f.endswith(".wav")])

    for fname in wav_files:
        speaker_id, label = parse_cremad_file(fname)
        if speaker_id is None:
            continue

        path = os.path.join(data_path, fname)
        feat = extract_logmel_3ch(
            path,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            max_len=max_len,
            use_speed_perturb=False,  # keep cache deterministic
        )

        X.append(feat)
        y.append(label)
        speakers.append(speaker_id)
        files.append(fname)

    X = np.array(X, dtype=np.float32)   # (N, 3, 128, T)
    y = np.array(y, dtype=np.int64)
    speakers = np.array(speakers, dtype=np.int64)
    files = np.array(files)

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