
import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("models/build_failure_model.pkl")

    sample = pd.DataFrame([{
        "commit_count": 3,
        "changed_files": 4,
        "lines_added": 40,
        "lines_deleted": 10,
        "test_count": 60,
        "failed_tests": 0,
        "build_duration": 150,
        "previous_failures": 0
    }])

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]

def test_model_probability():
    model = joblib.load("models/build_failure_model.pkl")

    sample = pd.DataFrame([{
        "commit_count": 3,
        "changed_files": 4,
        "lines_added": 40,
        "lines_deleted": 10,
        "test_count": 60,
        "failed_tests": 0,
        "build_duration": 150,
        "previous_failures": 0
    }])

    probability = model.predict_proba(sample)[0][1]

    assert 0 <= probability <= 1
