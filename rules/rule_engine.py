import logging
from typing import List, Set, Dict

# Configure logging
logger = logging.getLogger(__name__)

class RuleEngineError(Exception):
    """Custom exception for rule engine errors"""
    pass

# Define red flag symptoms with severity levels and recommendations
RED_FLAGS: Dict[str, Dict[str, str]] = {
    "chest pain": {
        "severity": "high",
        "recommendation": "May indicate heart problems. Seek immediate medical attention."
    },
    "shortness of breath": {
        "severity": "high",
        "recommendation": "Could be respiratory or cardiac related. Seek immediate care."
    },
    "severe headache": {
        "severity": "high",
        "recommendation": "If sudden and severe, could indicate serious conditions. Seek emergency care."
    },
    "unconsciousness": {
        "severity": "critical",
        "recommendation": "Medical emergency. Call emergency services immediately."
    },
    "vision loss": {
        "severity": "high",
        "recommendation": "Could indicate serious neurological issues. Seek immediate care."
    },
    "difficulty speaking": {
        "severity": "high",
        "recommendation": "Could indicate stroke. Seek emergency care."
    },
    "severe abdominal pain": {
        "severity": "high",
        "recommendation": "May indicate serious internal conditions. Seek immediate care."
    },
    "coughing blood": {
        "severity": "high",
        "recommendation": "Could indicate serious respiratory issues. Seek immediate care."
    }
}

def get_red_flag_keywords() -> Set[str]:
    """
    Get the set of red flag keywords.
    
    Returns:
        Set[str]: Set of red flag symptom keywords
    """
    return set(RED_FLAGS.keys())

def check_red_flags(symptoms: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Check if any high-risk (red flag) symptoms are present.

    Parameters:
        symptoms (List[str]): List of symptom strings.

    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary containing:
            - 'detected': List of dictionaries with detected red flags and their details
            - 'severity': Highest severity level found ('critical', 'high', or 'none')

    Raises:
        RuleEngineError: If symptom checking fails
    """
    if not isinstance(symptoms, list):
        raise RuleEngineError("Invalid input: symptoms must be a list")

    try:
        detected_flags = []
        max_severity = "none"
        
        # Normalize symptoms for comparison
        normalized_text = " ".join(s.lower() for s in symptoms)
        
        # Check for each red flag
        for flag, details in RED_FLAGS.items():
            if flag in normalized_text:
                detected_flags.append({
                    "symptom": flag,
                    "severity": details["severity"],
                    "recommendation": details["recommendation"]
                })
                
                # Update max severity
                if details["severity"] == "critical" or (
                    details["severity"] == "high" and max_severity != "critical"
                ):
                    max_severity = details["severity"]
        
        logger.info(f"Detected {len(detected_flags)} red flag symptoms")
        if detected_flags:
            logger.warning(f"Red flags found with max severity: {max_severity}")
            
        return {
            "detected": detected_flags,
            "severity": max_severity
        }

    except Exception as e:
        logger.error(f"Error checking red flags: {str(e)}")
        raise RuleEngineError(f"Failed to check red flags: {str(e)}")
