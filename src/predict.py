
import pandas as pd
import joblib

# Step 1: Load the trained model
model = joblib.load("models/build_failure_model.pkl")

# Step 2: Create a new build
new_build = pd.DataFrame([{
    "commit_count": 8,
    "changed_files": 15,
    "lines_added": 400,
    "lines_deleted": 120,
    "test_count": 100,
    "failed_tests": 0,
    "build_duration": 500,
    "previous_failures": 4
}])

# Step 3: Predict build status
prediction = model.predict(new_build)[0]

# Step 4: Calculate failure probability
failure_probability = model.predict_proba(new_build)[0][1]

# Step 5: Display the result
print("Build Failure Prediction")
print("----------------------")

if prediction == 1:
    print("Predicted Status: FAIL")
else:
    print("Predicted Status: PASS")

print("Failure Probability:",
      round(failure_probability * 100, 2), "%")
