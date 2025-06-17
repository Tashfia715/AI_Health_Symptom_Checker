import logging
from typing import Any, Optional
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Configure logging
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

def load_model(model_path: str = "models/symptom_classifier.pkl") -> Pipeline:
    """
    Load the trained model pipeline.

    Parameters:
        model_path (str): Path to the saved model.

    Returns:
        Pipeline: Trained pipeline model.

    Raises:
        PredictionError: If model loading fails
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise PredictionError(f"Failed to load model: {str(e)}")

def load_label_encoder(encoder_path: str = "models/label_encoder.pkl") -> LabelEncoder:
    """
    Load the saved label encoder.

    Parameters:
        encoder_path (str): Path to the saved LabelEncoder.

    Returns:
        LabelEncoder: Fitted label encoder.

    Raises:
        PredictionError: If encoder loading fails
    """
    try:
        encoder = joblib.load(encoder_path)
        logger.info(f"Label encoder loaded successfully from {encoder_path}")
        return encoder
    except Exception as e:
        logger.error(f"Failed to load label encoder from {encoder_path}: {str(e)}")
        raise PredictionError(f"Failed to load label encoder: {str(e)}")

def validate_prediction_input(model: Any, encoder: Any, text: str) -> None:
    """
    Validate inputs for prediction.

    Parameters:
        model: Model object to validate
        encoder: Encoder object to validate
        text (str): Input text to validate

    Raises:
        PredictionError: If validation fails
    """
    if not isinstance(text, str) or not text.strip():
        raise PredictionError("Input text must be a non-empty string")
    
    if not hasattr(model, 'predict'):
        raise PredictionError("Invalid model: missing predict method")
    
    if not hasattr(encoder, 'inverse_transform'):
        raise PredictionError("Invalid encoder: missing inverse_transform method")

def predict_condition(model: Pipeline, encoder: LabelEncoder, user_input: str) -> str:
    """
    Predict a medical condition from user input.

    Parameters:
        model (Pipeline): Trained model pipeline
        encoder (LabelEncoder): Trained LabelEncoder
        user_input (str): Symptom description from user

    Returns:
        str: Predicted condition label

    Raises:
        PredictionError: If prediction fails
    """
    try:
        # Validate inputs
        validate_prediction_input(model, encoder, user_input)
        
        # Make prediction
        pred_encoded = model.predict([user_input])[0]
        logger.debug(f"Raw prediction (encoded): {pred_encoded}")
        
        # Decode prediction
        prediction = encoder.inverse_transform([pred_encoded])[0]
        logger.info(f"Predicted condition: {prediction}")
        
        return prediction

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise PredictionError(f"Failed to make prediction: {str(e)}")
