import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_excel("fraud_ml_dataset.xlsx")

# Convert text columns to numbers
encoder = LabelEncoder()

df["Transaction Type"] = encoder.fit_transform(df["Transaction Type"])
df["Transaction Frequency"] = encoder.fit_transform(df["Transaction Frequency"])
df["Fraud Flag"] = encoder.fit_transform(df["Fraud Flag"])

# Features and target
X = df[["Transaction Amount",
        "Transaction Type",
        "Transaction Frequency",
        "Risk Score"]]

y = df["Fraud Flag"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
