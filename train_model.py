import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("--- Step 1: Loading REAL World Dataset ---")
# Load the CSV we just downloaded
if not os.path.exists('insurance.csv'):
    print("Error: insurance.csv not found!")
    exit()

df = pd.read_csv('insurance.csv')

# 2. Pre-processing Real Data
print("--- Step 2: Pre-processing Categorical Data ---")
# We need to turn words like 'male' and 'southwest' into numbers
encoders = {}
for column in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le
    print(f"Encoded {column}: {list(le.classes_)}")

# Separate features (X) and target (y)
X = df.drop('charges', axis=1)
y = df['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training the Model
print("--- Step 3: Training on Real Data ---")
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation (The Real Test!)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Step 4: REAL WORLD Results ---")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Model Accuracy (R2 Score): {r2*100:.2f}%")
print("\nNotice how the accuracy dropped from 99%? This is much more realistic!")

# 5. Save everything
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/insurance_model.pkl')
joblib.dump(encoders, 'models/encoders.pkl')

print("\nSuccess! Real-world model and encoders saved.")
