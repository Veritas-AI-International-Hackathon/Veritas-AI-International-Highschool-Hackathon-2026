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
# metadata 
#______print(statlog_german_credit_data.metadata) 
# variable information 
#_____print(statlog_german_credit_data.variables) 

# 1. Clean the target (y): 
# The dataset has classes: 1 = Good credit risk, 2 = Bad credit risk.
# Let's map it so 1 = Good (1) and 2 = Bad (0).
y_binary = y['class'].map({1: 1, 2: 0})

# 2. Encode all categorical features in X:
# pd.get_dummies automatically detects text columns (like A11, A12) 
# and converts them into separate 0/1 binary indicator columns.
X_encoded = pd.get_dummies(X, drop_first=True)

print(f"Original X shape: {X.shape}")
print(f"Encoded X shape (after turning text into numbers): {X_encoded.shape}")
print(f"Target distribution (1 = Good, 0 = Bad):\n{y_binary.value_counts()}")

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
print("--- Model 1 Performance ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.1%}")
print(f"Precision: {precision_score(y_test, y_pred):.1%} (of those predicted 'good', how many were actually good)")
print(f"Recall:    {recall_score(y_test, y_pred):.1%} (of actual 'good' credit risks, how many did we catch)")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

#bank cost rating

bank_cost = 5 * ((y_pred == 1) & (y_test == 0).values).sum() + ((y_pred == 0) & (y_test == 1).values).sum()
print(f"Bank cost:     {bank_cost} (5 x bad risks approved + 1 x good risks rejected; lower is better)")