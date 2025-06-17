import logging
import sys
from models.predict import load_model, load_label_encoder, predict_condition, PredictionError
from preprocessing.symptom_parser import extract_symptoms, SymptomParserError
from rules.rule_engine import check_red_flags, RuleEngineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_test_case(description: str, symptoms: str) -> None:
    """
    Run a single test case through the system.

    Parameters:
        description (str): Test case description
        symptoms (str): Symptom description to test
    """
    print(f"\nTesting: {description}")
    print(f"Input: {symptoms}")
    print("-" * 50)

    try:
        # Extract symptoms
        extracted = extract_symptoms(symptoms)
        print(f"Extracted Symptoms: {', '.join(extracted)}")

        # Check red flags
        flags = check_red_flags(extracted)
        if flags["detected"]:
            print("\nRed Flags Detected:")
            for flag in flags["detected"]:
                print(f"• {flag['symptom']}: {flag['recommendation']}")
        else:
            print("\nNo red flags detected")

        # Make prediction
        model = load_model()
        encoder = load_label_encoder()
        prediction = predict_condition(model, encoder, symptoms)
        print(f"\nPredicted Condition: {prediction}")

        print("\nTest completed successfully")

    except (PredictionError, SymptomParserError, RuleEngineError) as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n Test failed: {str(e)}")

def main():
    """Run a series of test cases through the system."""
    print("\n AI Symptom Checker - System Test")
    print("=" * 50)

    test_cases = [
        (
            "Basic Symptoms",
            "I have a headache and fever"
        ),
        (
            "Red Flag Symptoms",
            "I'm experiencing severe chest pain and shortness of breath"
        ),
        (
            "Multiple Symptoms",
            "I've been having a cough, runny nose, and sore throat for the past few days"
        ),
        (
            "Complex Description",
            "For the past week, I've been feeling tired, having trouble sleeping, "
            "and experiencing muscle aches throughout my body"
        ),
        (
            "Empty Input Test",
            ""
        )
    ]

    try:
        for description, symptoms in test_cases:
            run_test_case(description, symptoms)
            print("\n" + "=" * 50)

    except KeyboardInterrupt:
        print("\n\n⚠  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        print(f"\n An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
