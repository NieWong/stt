import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from mltu.configs import BaseModelConfigs
from WavToTextModel import WavToTextModel
from AudioProcessing import predict_from_audio

# Load configurations
configs = BaseModelConfigs.load("/home/sukhe/Documents/stt/Models/05_sound_to_text/202302051936/configs.yaml")
print(f"Loaded Configs: {configs}")

# Parameters for recording
SAMPLE_RATE = 16000  # Common sampling rate for speech models
DURATION = 5  # Duration in seconds

def record_audio(filename="mic_input.wav", duration=DURATION, sample_rate=SAMPLE_RATE):
    """Records audio from the microphone."""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is complete
    write(filename, sample_rate, audio)  # Save as WAV file
    print(f"Recording saved to {filename}")
    return filename

if __name__ == "__main__":
    # Verify the model path
    model_path = f"{configs.model_path}/model.onnx"
    print(f"Final Model Path: {model_path}")

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    # Initialize the model
    model = WavToTextModel(
        model_path=model_path,
        char_list=configs.vocab
    )

    # Record audio from the microphone
    audio_path = record_audio()

    # Call the function to predict text
    predict_from_audio(audio_path, model, configs)
