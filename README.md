# AI Health Symptom Checker

An intelligent system that analyzes user-described symptoms to suggest possible medical conditions and identify potential red flags.

## Features

-  Natural Language Processing for symptom extraction
-  Machine Learning-based condition prediction
-  Red flag symptom detection
-  Explainable AI integration
-  High accuracy symptom classification

## Project Structure

```
AI_Health_Symptom_Checker/
├── README.md
├── app
│   └── app.py
├── app.py
├── dataset
│   └── train.csv
├── explainability
│   └── explain_model.py
├── helpers
│   └── helpers.py
├── models
│   ├── __pycache__
│   │   ├── predict.cpython-312.pyc
│   │   └── train_classifier.cpython-312.pyc
│   ├── label_encoder.pkl
│   ├── predict.py
│   ├── symptom_classifier.pkl
│   └── train_classifier.py
├── preprocessing
│   ├── __pycache__
│   │   ├── clean_data.cpython-312.pyc
│   │   └── symptom_parser.cpython-312.pyc
│   ├── clean_data.py
│   └── symptom_parser.py
├── requirements.txt
├── rules
│   ├── __pycache__
│   │   └── rule_engine.cpython-312.pyc
│   └── rule_engine.py
├── test_system.py
└── trainer.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required MedSpaCy models:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the CLI application:
```bash
python app.py
```

2. Enter your symptoms when prompted
3. The system will:
   - Extract key symptoms from your description
   - Predict possible conditions
   - Check for any red flag symptoms
   - Provide appropriate warnings if necessary

## Model Training

To train a new model:

1. Ensure your training data is in the correct format in `dataset/train.csv`
2. Run the training script:
```bash
python trainer.py
```

The script will:
- Clean and preprocess the data
- Handle class imbalance using SMOTE
- Perform stratified k-fold cross-validation
- Train an XGBoost classifier with optimized parameters
- Evaluate model performance with detailed metrics:
  - Cross-validation accuracy with standard deviation
  - Per-class precision, recall, and F1-scores
  - Class distribution analysis
- Save the trained model and label encoder

The training process automatically handles rare conditions and class imbalance, ensuring robust performance across all symptom categories. The system uses SMOTE to generate synthetic samples for underrepresented conditions, and k-fold cross-validation to provide reliable performance estimates.

## Technical Details

- **Symptom Extraction**: Uses MedSpaCy for clinical NLP
- **Classification**: 
  - XGBoost classifier with TF-IDF vectorization
  - SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced classes
  - Stratified K-fold cross-validation for robust evaluation
  - Handles rare conditions with configurable minimum sample threshold
- **Validation**: 
  - Cross-validation accuracy with standard deviation
  - Detailed classification metrics per condition
  - Class distribution analysis
- **Red Flag Detection**: 
  - Rule-based system for urgent symptom identification
  - Severity levels (high/critical) with specific recommendations
  - Comprehensive coverage of emergency symptoms

## Important Notes

This system is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for medical concerns.

## Dependencies

- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms and preprocessing
- xgboost: Gradient boosting implementation
- medspacy: Clinical natural language processing
- joblib: Model persistence and loading
- shap: Model explainability and feature importance
- imbalanced-learn: SMOTE implementation for handling imbalanced classes
- numpy: Numerical computing and array operations
- tqdm: Progress bars for long-running operations
- spacy: Core NLP functionality (required by medspacy)

## Future Improvements

- Web interface implementation
- Additional symptom extraction methods
- Enhanced red flag detection
- Integration with medical knowledge bases
- Multi-language support
