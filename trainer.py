import logging
import sys
from pathlib import Path
from preprocessing.clean_data import load_and_clean, DataCleaningError
from models.train_classifier import train_model, ModelTrainingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def main():
    """
    Main training script that:
    1. Loads and cleans the training data
    2. Trains the model
    3. Saves the model and encoder
    """
    try:
        print("\n AI Symptom Checker - Model Training")
        print("=" * 50)

        # Setup directories
        setup_directories()
        
        # Load and clean data
        print("\n Loading and cleaning training data...")
        try:
            df = load_and_clean("dataset/train.csv")
            print(f" Loaded {len(df)} training samples")
        except DataCleaningError as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            print(f"\n Error cleaning data: {str(e)}")
            sys.exit(1)

        # Train model
        print("\n Training model...")
        try:
            train_model(
                df=df,
                model_path="models/symptom_classifier.pkl",
                encoder_path="models/label_encoder.pkl",
                test_size=0.2,
                max_features=5000
            )
            print("\n Training completed successfully!")
            
        except ModelTrainingError as e:
            logger.error(f"Model training failed: {str(e)}")
            print(f"\n Error training model: {str(e)}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        print(f"\n An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
