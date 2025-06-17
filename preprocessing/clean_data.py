import logging
from typing import Optional
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

class DataCleaningError(Exception):
    """Custom exception for data cleaning errors"""
    pass

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> None:
    """
    Validate DataFrame structure and content.

    Parameters:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names

    Raises:
        DataCleaningError: If validation fails
    """
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise DataCleaningError(f"Missing required columns: {', '.join(missing_cols)}")

    # Check for empty DataFrame
    if df.empty:
        raise DataCleaningError("DataFrame is empty")

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV and cleans the data by removing missing values,
    duplicates, and validating required columns.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.

    Raises:
        DataCleaningError: If data loading or cleaning fails
    """
    required_columns = ["qtype", "Question", "Answer"]

    try:
        # Load data
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate DataFrame
        validate_dataframe(df, required_columns)
        
        # Record initial size
        initial_size = len(df)
        logger.info(f"Initial dataset size: {initial_size} rows")

        # Drop rows with missing values
        df = df.dropna(subset=required_columns)
        missing_dropped = initial_size - len(df)
        logger.info(f"Dropped {missing_dropped} rows with missing values")

        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_dropped = initial_size - missing_dropped - len(df)
        logger.info(f"Dropped {duplicates_dropped} duplicate rows")

        # Basic text cleaning
        for col in ["Question", "Answer"]:
            df[col] = df[col].str.strip()

        logger.info(f"Final dataset size: {len(df)} rows")
        return df

    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {filepath}")
        raise DataCleaningError("The CSV file is empty")
    
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {filepath}: {str(e)}")
        raise DataCleaningError(f"Failed to parse CSV file: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise DataCleaningError(f"Data cleaning failed: {str(e)}")

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame to save
        output_path (str): Path where to save the cleaned data

    Raises:
        DataCleaningError: If saving fails
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving cleaned data: {str(e)}")
        raise DataCleaningError(f"Failed to save cleaned data: {str(e)}")
