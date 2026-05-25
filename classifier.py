# Data Classification Using AI - KNN
# DecodeLabs Industrial Training | Batch 2026
# Developer: Mayar Yasser

# STEP 1: IMPORT LIBRARIES
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 2: LOAD DATA
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print(f"Total samples: {X.shape[0]}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}\n")

# STEP 3: FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 4: TRAIN-TEST SPLIT (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")

# STEP 5: TRAIN THE MODEL (KNN)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Model trained successfully!\n")

# STEP 6: PREDICTIONS
predictions = model.predict(X_test)

# STEP 7: RESULTS
print("=" * 40)
print("CLASSIFICATION REPORT:")
print("=" * 40)
print(classification_report(y_test, predictions,
      target_names=iris.target_names))

# STEP 8: CONFUSION MATRIX
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cmap='Blues')
plt.title('Confusion Matrix - Mayar Yasser')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved!")