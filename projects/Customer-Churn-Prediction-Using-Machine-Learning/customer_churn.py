# CUSTOMER CHURN PREDICTION USING MACHINE LEARNING

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_excel("churn_dataset.xlsx")

print("Dataset Preview:")
print(df.head())

# Encode categorical variables
label_encoder = LabelEncoder()

df["Contract_Type"] = label_encoder.fit_transform(df["Contract_Type"])
df["Payment_Method"] = label_encoder.fit_transform(df["Payment_Method"])

# Remove Customer_ID from model training
X = df.drop(["Customer_ID", "Churn"], axis=1)
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Retained", "Churned"],
    yticklabels=["Retained", "Churned"]
)

plt.title("Customer Churn Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")

# Save model
joblib.dump(model, "churn_model.pkl")

print("Model saved as churn_model.pkl")
