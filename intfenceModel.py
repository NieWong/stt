import typing
import numpy as np
import os
from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
import onnxruntime as ort  # ONNX runtime

# Load configurations
configs = BaseModelConfigs.load("/home/sukhe/Documents/stt/Models/05_sound_to_text/202302051936/configs.yaml")
print(f"Loaded Configs: {configs}")

class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

        # Check if model path is valid
        if not model_path:
            raise ValueError("Model path is empty or not set.")

        print(f"Loading ONNX model from: {model_path}")

        # Load ONNX model using ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, data: np.ndarray):
        """Predict text from input data."""
        data_pred = np.expand_dims(data, axis=0).astype(np.float32)
        preds = self.session.run([self.output_name], {self.input_name: data_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def predict_from_audio(audio_path: str, model: WavToTextModel, configs: BaseModelConfigs):
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

    # Provide the audio file path for testing
    audio_path = "path/to/custom_audio.wav"

    # Call the function to predict text
    predict_from_audio(audio_path, model, configs)
