import shap
import joblib
import numpy as np
import pandas as pd

def explain_prediction(
    model_path: str,
    input_text: str,
    top_n_words: int = 10
) -> None:
    """
    Explains a model prediction using SHAP.

    Parameters:
        model_path (str): Path to the trained model (.pkl).
        input_text (str): User input to explain.
        top_n_words (int): Number of most influential words to show.
    """
    # Load model
    model_pipeline = joblib.load(model_path)

    # Split pipeline into vectorizer and model
    vectorizer = model_pipeline.named_steps["tfidf"]
    model = model_pipeline.named_steps["clf"]

    # Transform input
    text_vector = vectorizer.transform([input_text])

    # Initialize SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(text_vector)

    # Get feature importance
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = shap_values.values[0]
    top_indices = np.argsort(np.abs(scores))[::-1][:top_n_words]

    print("\n Top Influential Words:")
    for i in top_indices:
        print(f"  {feature_names[i]}: {scores[i]:.4f}")

    # Optional: SHAP bar chart
    shap.plots.bar(shap_values, max_display=top_n_words)
