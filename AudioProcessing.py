import numpy as np
from mltu.preprocessors import WavReader
from mltu.configs import BaseModelConfigs

def predict_from_audio(audio_path: str, model, configs: BaseModelConfigs):
    """Preprocess audio and predict text."""
    # Generate a spectrogram from the audio file
    spectrogram = WavReader.get_spectrogram(
        audio_path,
        frame_length=configs.frame_length,
        frame_step=configs.frame_step,
        fft_length=configs.fft_length
    )

    # Pad the spectrogram to match input shape
    padded_spectrogram = np.pad(
        spectrogram,
        ((0, configs.max_spectrogram_length - spectrogram.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0
    )

    # Predict text from audio
    predicted_text = model.predict(padded_spectrogram)
    print(f"Predicted Text: {predicted_text}")
