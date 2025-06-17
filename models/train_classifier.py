import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logger = logging.getLogger(__name__)

class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass

def validate_training_data(df: pd.DataFrame) -> None:
    """
    Validate training data structure and content.

    Parameters:
        df (pd.DataFrame): Training data to validate

    Raises:
        ModelTrainingError: If validation fails
    """
    required_columns = ["Question", "Answer"]
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ModelTrainingError(f"Missing required columns: {', '.join(missing_cols)}")

    # Check for empty DataFrame
    if df.empty:
        raise ModelTrainingError("Training data is empty")

    # Log class distribution
    class_counts = df["Answer"].value_counts()
    logger.info("Class distribution:")
    for class_name, count in class_counts.items():
        logger.info(f"{class_name}: {count} samples")

def prepare_data(df: pd.DataFrame, min_samples: int = 1) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare data for training, handling rare classes.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        min_samples (int): Minimum samples per class (default: 1)

    Returns:
        Tuple[pd.Series, pd.Series]: Features and labels
    """
    # Filter classes with fewer than min_samples
    class_counts = df["Answer"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_filtered = df[df["Answer"].isin(valid_classes)]
    
    logger.info(f"Using {len(valid_classes)} classes with ≥{min_samples} sample(s)")
    
    if len(df_filtered) < len(df):
        removed = len(df) - len(df_filtered)
        logger.warning(f"Removed {removed} samples from rare classes")
    
    return df_filtered["Question"], df_filtered["Answer"]

def create_model_pipeline(max_features: int = 5000) -> ImbPipeline:
    """
    Create the model pipeline with TF-IDF, SMOTE, and XGBoost.

    Parameters:
        max_features (int): Maximum number of features for TF-IDF

    Returns:
        ImbPipeline: Configured model pipeline with SMOTE
    """
    return ImbPipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            strip_accents='unicode',
            lowercase=True
        )),
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        ))
    ])

def train_model(
    df: pd.DataFrame,
    model_path: str = "models/symptom_classifier.pkl",
    encoder_path: str = "models/label_encoder.pkl",
    test_size: float = 0.2,
    max_features: int = 5000,
    n_splits: int = 5,
    min_samples: int = 1
) -> None:
    """
    Train a symptom classifier using TF-IDF, SMOTE, and XGBoost.
    Uses stratified k-fold cross-validation for evaluation.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Question' and 'Answer' columns
        model_path (str): Path to save the trained model
        encoder_path (str): Path to save the label encoder
        test_size (float): Proportion of data to use for testing
        max_features (int): Maximum number of features for TF-IDF
        n_splits (int): Number of folds for cross-validation
        min_samples (int): Minimum samples required per class

    Raises:
        ModelTrainingError: If training fails
    """
    try:
        # Validate input data
        validate_training_data(df)
        logger.info("Data validation passed")

        # Prepare data
        X, y = prepare_data(df, min_samples)
        logger.info(f"Prepared dataset with {len(X)} samples")

        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Cross-validation
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            # Create and train pipeline
            pipeline = create_model_pipeline(max_features)
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = pipeline.predict(X_val)
            fold_acc = accuracy_score(y_val, y_pred)
            cv_scores.append(fold_acc)
            
            logger.info(f"Fold {fold}/{n_splits} - Accuracy: {fold_acc:.3f}")

        # Train final model on full dataset
        final_pipeline = create_model_pipeline(max_features)
        final_pipeline.fit(X, y_encoded)

        # Print results
        mean_acc = np.mean(cv_scores)
        std_acc = np.std(cv_scores)
        print(f"\n✅ Cross-validation Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        
        # Final evaluation on full dataset
        y_pred = final_pipeline.predict(X)
        print("\n📊 Final Classification Report:")
        print(classification_report(
            y_encoded, y_pred,
            target_names=label_encoder.classes_,
            labels=np.unique(y_encoded)
        ))

        # Save model and encoder
        joblib.dump(final_pipeline, model_path)
        joblib.dump(label_encoder, encoder_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Label encoder saved to: {encoder_path}")
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"🔤 Label encoder saved to: {encoder_path}")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise ModelTrainingError(f"Model training failed: {str(e)}")
