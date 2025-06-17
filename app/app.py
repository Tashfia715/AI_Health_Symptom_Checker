from models.predict import load_model, predict_condition
from preprocessing.symptom_parser import extract_symptoms
from rules.rule_engine import check_red_flags

def main():
    model = load_model()

    print("AI Symptom Checker\nType 'exit' to quit.\n")

    while True:
        user_input = input("Describe your symptoms: ")
        if user_input.lower() == "exit":
            break

        symptoms = extract_symptoms(user_input)
        prediction = predict_condition(model, user_input)
        urgent = check_red_flags(symptoms)

        print(f"Likely Condition: {prediction}")
        print(f"Extracted Symptoms: {symptoms}")
        if urgent:
            print(" Warning: Your symptoms may indicate a serious condition.")
        print("-" * 40)

if __name__ == "__main__":
    main()
