import logging
from typing import Optional, Tuple
import sys
from models.predict import load_model, load_label_encoder, predict_condition
from preprocessing.symptom_parser import extract_symptoms
from rules.rule_engine import check_red_flags

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_input(text: str) -> bool:
    """
    Validate user input for minimum length and content.
    
    Parameters:
        text (str): User input text
        
    Returns:
        bool: True if input is valid, False otherwise
    """
    if not text or len(text.strip()) < 3:
        return False
    return True

def initialize_system() -> Tuple[Optional[object], Optional[object]]:
    """
    Initialize the symptom checker system by loading model and encoder.
    
    Returns:
        Tuple[Optional[object], Optional[object]]: Loaded model and encoder, or None if loading fails
    """
    try:
        model = load_model()
        encoder = load_label_encoder()
        logger.info("System initialized successfully")
        return model, encoder
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return None, None

def process_symptoms(model: object, encoder: object, user_input: str) -> Tuple[str, list, dict]:
    """
    Process user symptoms and return prediction, extracted symptoms, and urgency details.
    
    Parameters:
        model: Trained model pipeline
        encoder: Label encoder
        user_input (str): User's symptom description
        
    Returns:
        Tuple[str, list, dict]: Prediction, extracted symptoms, and urgency details
    """
    try:
        # Extract symptoms using NLP
        symptoms = extract_symptoms(user_input)
        logger.info(f"Extracted symptoms: {symptoms}")

        # Predict condition
        prediction = predict_condition(model, encoder, user_input)
        logger.info(f"Predicted condition: {prediction}")

        # Check for urgent symptoms
        red_flags = check_red_flags(symptoms)
        if red_flags["detected"]:
            logger.warning(f"Red flag symptoms detected with severity: {red_flags['severity']}")

        return prediction, symptoms, red_flags
    except Exception as e:
        logger.error(f"Error processing symptoms: {str(e)}")
        raise

def main():
    """Main application loop"""
    try:
        # Initialize system
        model, encoder = initialize_system()
        if not model or not encoder:
            logger.error("Failed to initialize system. Exiting.")
            sys.exit(1)

        print("\n🤖 AI Symptom Checker")
        print("Type 'exit' to quit.\n")
        print("⚠️  DISCLAIMER: This is for educational purposes only. Always consult a healthcare professional.\n")

        while True:
            try:
                user_input = input("Describe your symptoms: ").strip()
                
                if user_input.lower() == "exit":
                    print("\nThank you for using AI Symptom Checker. Take care! 👋")
                    break

                if not validate_input(user_input):
                    print("❌ Please provide a more detailed description of your symptoms.")
                    continue

                # Process symptoms and get results
                prediction, symptoms, red_flags = process_symptoms(model, encoder, user_input)

                # Output results
                print("\n🩺 Analysis Results:")
                print("-" * 50)
                print(f"🔍 Extracted Symptoms: {', '.join(symptoms) if symptoms else 'No specific symptoms detected'}")
                print(f"📋 Likely Condition: {prediction}")
                
                # Display red flag warnings if any
                if red_flags["detected"]:
                    print("\n⚠️  WARNING: Serious symptoms detected!")
                    print("🏥 Medical Attention Required:")
                    for flag in red_flags["detected"]:
                        print(f"\n• {flag['symptom'].title()}:")
                        print(f"  Severity: {flag['severity'].upper()}")
                        print(f"  {flag['recommendation']}")
                
                print("\n⚕️  IMPORTANT:")
                print("1. This is not a substitute for professional medical advice")
                print("2. If symptoms persist or worsen, consult a healthcare provider")
                print("3. In case of emergency, call your local emergency services")
                print("-" * 50 + "\n")

            except Exception as e:
                logger.error(f"Error processing user input: {str(e)}")
                print("❌ An error occurred while processing your symptoms. Please try again.")

    except KeyboardInterrupt:
        print("\n\nExiting gracefully... Take care! 👋")
    except Exception as e:
        logger.critical(f"Critical error in main loop: {str(e)}")
        print("\n❌ A critical error occurred. Please check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
