import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("parkinsons_updrs.data")

# Define features and target variable
selected_features = ["RPDE", "DFA", "HNR", "Shimmer:APQ5", "PPE"]
X = df[selected_features]
y = df["total_UPDRS"].apply(lambda x: 1 if x > df["total_UPDRS"].median() else 0)  # Binary classification

# Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model and scaler
joblib.dump(model, "parkinsons_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
