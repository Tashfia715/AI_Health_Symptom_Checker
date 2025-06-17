import logging
from typing import List, Optional
import medspacy

# Configure logging
logger = logging.getLogger(__name__)

class SymptomParserError(Exception):
    """Custom exception for symptom parsing errors"""
    pass

def initialize_nlp() -> Optional[object]:
    """
    Initialize the medspaCy NLP pipeline.
    
    Returns:
        Optional[object]: Loaded NLP pipeline or None if loading fails
    """
    try:
        nlp = medspacy.load()
        logger.info("MedspaCy NLP pipeline initialized successfully")
        return nlp
    except Exception as e:
        logger.error(f"Failed to initialize MedspaCy: {str(e)}")
        return None

# Initialize NLP pipeline
nlp = initialize_nlp()
if nlp is None:
    raise SymptomParserError("Failed to initialize NLP pipeline")

def extract_symptoms(text: str) -> List[str]:
    """
    Extracts symptom-like entities using medspaCy's clinical NLP pipeline.

    Parameters:
        text (str): User symptom description.

    Returns:
        List[str]: Extracted symptom terms.

    Raises:
        SymptomParserError: If symptom extraction fails.
    """
    if not text or not isinstance(text, str):
        raise SymptomParserError("Invalid input: text must be a non-empty string")

    try:
        # Process text with NLP pipeline
        doc = nlp(text)
        
        # Extract relevant entities
        symptoms = [
            ent.text for ent in doc.ents 
            if ent._.is_symptom or ent.label_.lower() in {"problem", "condition"}
        ]
        
        # Remove duplicates while preserving order
        unique_symptoms = list(dict.fromkeys(symptoms))
        
        logger.info(f"Extracted {len(unique_symptoms)} unique symptoms from text")
        return unique_symptoms

    except Exception as e:
        logger.error(f"Error during symptom extraction: {str(e)}")
        raise SymptomParserError(f"Failed to extract symptoms: {str(e)}")
