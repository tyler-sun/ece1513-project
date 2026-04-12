import os
import torch
import numpy as np

from load_data import (
    extract_logmel_3ch,
)
from train import NUM_CLASSES, SER_CNN_Attention

WEIGHTS_DIR = "outputs"
emotion_map_reverse = {
    0: "ANG",
    1: "DIS",
    2: "FEA",
    3: "HAP",
    4: "NEU",
    5: "SAD",
}

# Simple script which takes user inputted file and runs model on one .wav file sample for a prediction
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and weights from training
    model = SER_CNN_Attention(NUM_CLASSES).to(device)
    weights_path = os.path.join(WEIGHTS_DIR, f"best.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
        print("Loaded model weights from", os.path.join(WEIGHTS_DIR, f"best.pt"))
    else:
        print(f"Model weights not found at {weights_path}")
        return

    # Load normalization statistics file from training
    norm_path = os.path.join(WEIGHTS_DIR, "normalization_stats.npz")
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        mean = norm_data["mean"]
        std = norm_data["std"]
        print(f"Loaded normalization stats from {norm_path}")
    else:
        print(f"Normalization stats not found at {norm_path}")
        return

    print("\nPlease type the path to a .wav file to run the model, or type 'q' to quit:")
    while True:
        file_input = input("Enter file path: ")
        if file_input.lower() == 'q' or file_input.lower() == 'quit':
            break
        if not os.path.isfile(file_input) or not file_input.endswith('.wav'):
            print("Invalid file path. Please enter a valid .wav file path.")
            continue
        else:
            try:
                print("Reading audio from", file_input)
                audio_file = file_input
                # Extract features and normalize with training stats
                audio_features = extract_logmel_3ch(file_path=audio_file)
                audio_features = (audio_features - mean) / (std + 1e-8)
                
                audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(audio_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_label = torch.argmax(probabilities, dim=1).item()
                
                pred_idx = logits.argmax(dim=1).item()
                confidence = probabilities[0, pred_idx].item() * 100
                predicted_emotion = emotion_map_reverse.get(predicted_label, "Unknown")
                print(f"Predicted emotion: {predicted_emotion} with confidence of {confidence:.4f}%")
                continue
            
            except Exception as e:
                print(f"Error processing file: {e}")
                break


if __name__ == "__main__":
    main()