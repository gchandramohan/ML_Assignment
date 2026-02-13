# train_models.py (FAST VERSION)

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# -------------------------------
# Create folders
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("test_data", exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("Dry_Bean_Dataset.csv")
df = df.dropna()

# -------------------------------
# Encode target
# -------------------------------
le = LabelEncoder()
df["Class"] = le.fit_transform(df["Class"])

X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------
# Preprocessing BEFORE split
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Save test CSV (original labels)
# -------------------------------
original_labels = le.inverse_transform(y_test)

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["Class"] = original_labels
test_df.to_csv("test_data/dry_bean_test.csv", index=False)

# -------------------------------
# Train Models (NO GridSearch)
# -------------------------------

# 1 Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
pickle.dump(log_model, open("models/logistic.pkl", "wb"))

# 2 Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(X_train, y_train)
pickle.dump(dt_model, open("models/decision_tree.pkl", "wb"))

# 3 KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
pickle.dump(knn_model, open("models/knn.pkl", "wb"))

# 4 Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
pickle.dump(nb_model, open("models/naive_bayes.pkl", "wb"))

# 5 Random Forest (parallel)
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_model.fit(X_train, y_train)
pickle.dump(rf_model, open("models/random_forest.pkl", "wb"))

# 6 Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100)
gb_model.fit(X_train, y_train)
pickle.dump(gb_model, open("models/gradient_boost.pkl", "wb"))

print("All models trained successfully.")
