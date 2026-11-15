import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 

# 1. Clean the target (y): 
# The dataset has classes: 1 = Good credit risk, 2 = Bad credit risk.
# Let's map it so 1 = Good (1) and 2 = Bad (0).
y_binary = y['class'].map({1: 1, 2: 0})

# 2. Encode all categorical features in X:
# pd.get_dummies automatically detects text columns (like A11, A12) 
# and converts them into separate 0/1 binary indicator columns.
X_encoded = pd.get_dummies(X, drop_first=True)

#TRAINING THE MODEL


# 1. Split data into training (80%) and testing (20%) sets
# stratify=y_binary ensures both sets have the same ratio of good/bad credit risks
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Training set: {len(X_train)} applicants")
print(f"Test set: {len(X_test)} applicants\n")

# 2. Initialize and train the baseline model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Make predictions on the test set
y_pred = model.predict(X_test)

# 4. Evaluate performance
print("--- Baseline Model Performance ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.1%}")
print(f"Precision: {precision_score(y_test, y_pred):.1%} (of those predicted 'good', how many were actually good)")
print(f"Recall:    {recall_score(y_test, y_pred):.1%} (of actual 'good' credit risks, how many did we catch)")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

#code to test strength of strictness in target groups
import numpy as np
zone_tr = ((X_train['Attribute13'] <= 30) | X_train['Attribute2'].between(13, 24)).values
zone_te = ((X_test['Attribute13'] <= 30) | X_test['Attribute2'].between(13, 24)).values
bad_tr = (y_train == 0).values
bad, good = zone_te & (y_test == 0).values, zone_te & (y_test == 1).values
print(f"{'k':<5}{'Overall accuracy':<20}{'Bad approved':<16}{'Good rejected'}")
for k in [1, 2, 3, 5]:
    w = np.where(zone_tr & bad_tr, k, 1)   # extra weight on BAD-risk people in problem groups
    p = LogisticRegression(max_iter=1000).fit(X_train, y_train, sample_weight=w).predict(X_test)
    print(f"{k:<5}{accuracy_score(y_test, p):<20.1%}{p[bad].mean():<16.1%}{1 - p[good].mean():.1%}")