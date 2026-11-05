
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 1: Load the dataset
df = pd.read_csv("data/build_data.csv")

print("Dataset loaded successfully")
print(df.head())

# Step 2: Separate input features and target
X = df.drop(columns=["build_id", "build_status"])
y = df["build_status"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 4: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 5: Train the model
model.fit(X_train, y_train)

print("\nModel training completed")

# Step 6: Predict test results
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Display feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Step 9: Save the trained model
joblib.dump(model, "models/build_failure_model.pkl")

print("\nModel saved successfully")
import matplotlib.pyplot as plt

importance.plot(kind="bar", figsize=(10, 5))

plt.title("Build Failure Prediction - Feature Importance")
plt.xlabel("Build Features")
plt.ylabel("Importance Score")
plt.tight_layout()

plt.savefig("models/feature_importance.png")
plt.show()

print("Feature importance graph saved successfully")