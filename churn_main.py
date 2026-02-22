import pandas as pd
import numpy as np
import joblib  # This is the tool to save your model to a file

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Import your optimized utility script
# Make sure mesh_utils_optimized.py is in the same folder!

from ml_utils import mesh_utils_optimized as muo

# ==========================================
# 1. Data Loading & Cleaning
# ==========================================
print("Loading and cleaning data...")
df = pd.read_csv('Telco_Customer_Churn.csv')

# Fix TotalCharges (Convert to number, fill blanks with 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Drop ID
df = df.drop(columns=['customerID'])

# ==========================================
# 2. Feature Selection
# ==========================================
# Defining features to drop (Redundant or Useless)
useless_cols = [
    'gender',
    'PhoneService',
    'MultipleLines',
    'TotalCharges',  # Redundant with tenure
    'StreamingTV',
    'StreamingMovies'
]

# Create X and y
X = df.drop(columns=['Churn'] + useless_cols)
y = df['Churn'].map({'Yes': 1, 'No': 0}) # Target must be numeric

print(f"Features kept: {list(X.columns)}")

# ==========================================
# 3. Preprocessing Pipeline (The "Brain")
# ==========================================
# Identify which columns are numbers vs text
numeric_features = ['tenure', 'MonthlyCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# Create the ColumnTransformer
# This bundle says: "Scale the numbers, One-Hot Encode the text. Ignore the rest."
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Fit and Transform the data
print("\nRunning Preprocessing Pipeline...")
X_processed = preprocessor.fit_transform(X)

# ==========================================
# 4. Model Training (Auto-ML)
# ==========================================
# Find the best model using F1 Score (better for imbalance)
print("\nSearching for best model (optimizing for F1 Score)...")
result = muo.find_best_model(X_processed, y, "classification", "f1", n_iter=10)

print("-" * 40)
print(f"WINNER: {result['best_model_name']}")
print(f"CV F1 Score: {result['CV_score']:.4f}")
print(f"Test F1 Score: {result['Test_score']:.4f}")
print("-" * 40)

# ==========================================
# 5. Save for Deployment (The Critical Step)
# ==========================================
print("\nSaving files for Streamlit deployment...")

# Save the Best Model
best_model = result['trained_model']
joblib.dump(best_model, 'best_churn_model.pkl')

# Save the Preprocessor (So the app knows how to scale inputs)
joblib.dump(preprocessor, 'preprocessor.pkl')

print("SUCCESS! Files 'best_churn_model.pkl' and 'preprocessor.pkl' saved.")