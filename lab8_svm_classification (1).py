# Lab 8: Support Vector Machine (SVM) for Classification
# Dataset: Iris dataset (built into scikit-learn)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Class names:", iris.target_names)
print("Missing values:", np.isnan(X).sum())

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train SVM classifier
model = SVC(kernel="linear", C=1.0)
model.fit(X_train_scaled, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("SVM Classification - Iris Dataset")
plt.tight_layout()
plt.show()
