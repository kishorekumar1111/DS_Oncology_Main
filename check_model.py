import joblib
from pathlib import Path

MODEL_FILE = Path("output/ml/oncology_toxicity_risk_model.pkl")

model = joblib.load(MODEL_FILE)

print("=" * 60)
print("MODEL INPUT CHECK")
print("=" * 60)

print("\nModel type:")
print(type(model))

if hasattr(model, "feature_names_in_"):
    print("\nExpected columns:")
    for col in model.feature_names_in_:
        print("-", col)

    print("\nTotal columns:", len(model.feature_names_in_))
else:
    print("\nfeature_names_in_ not available")

if hasattr(model, "classes_"):
    print("\nClasses:")
    print(model.classes_)