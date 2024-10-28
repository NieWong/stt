import typing
import numpy as np
import onnxruntime as ort  # ONNX runtime
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], model_path: str, *args, **kwargs):
        # Pass model_path explicitly to parent class
        super().__init__(model_path=model_path, *args, **kwargs)
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
